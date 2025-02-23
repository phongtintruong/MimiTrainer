from typing import Optional, Union, List, Tuple
import typing as tp
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers import MimiModel, MimiConfig
from transformers.models.mimi.modeling_mimi import MimiSplitResidualVectorQuantizer, MimiOutput, MimiEncoderOutput, MimiModel, MimiResidualVectorQuantizer, MimiEuclideanCodebook, MimiVectorQuantization
from torch import nn
from transformers import PretrainedConfig
from transformers.utils import cached_file, WEIGHTS_NAME, SAFE_WEIGHTS_NAME
import safetensors.torch
from einops import rearrange, repeat


class MeomeoConfig(MimiConfig):

    model_type = "meomeo"

    def __init__(
        self, 
        acoustic_quantization_skipping_rate,
        acoustic_min_nq,
        semantic_quantization_skipping_rate,
        semantic_min_nq,
        hidden_prj_size=512, 
        output_prj_size=1024,
        **kwargs):
        super().__init__(**kwargs)
        self.hidden_prj_size = hidden_prj_size
        self.output_prj_size = output_prj_size
        self.acoustic_quantization_skipping_rate = acoustic_quantization_skipping_rate
        self.acoustic_min_nq = acoustic_min_nq
        self.semantic_quantization_skipping_rate = semantic_quantization_skipping_rate
        self.semantic_min_nq = semantic_min_nq

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config_dict, kwargs = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # Ensure we only pass expected parameters to MeomeoConfig
        return cls(**config_dict, **kwargs)

@dataclass
class MeomeoOutput(MimiOutput):
    """
    Args:
        audio_codes (`torch.LongTensor`  of shape `(batch_size, num_quantizers, codes_length)`, *optional*):
            Discret code embeddings computed using `model.encode`.
        audio_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*)
            Decoded audio values, obtained using the decoder part of Mimi.
        encoder_past_key_values (`Cache`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks) that can be used to speed up sequential decoding of the encoder transformer.
            This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            The model will output the same cache format that is fed as input.

            If `past_key_values` are used, the user can optionally input only the last `audio_values` or `audio_codes (those that don't
            have their past key value states given to this model).
        decoder_past_key_values (`Cache`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks) that can be used to speed up sequential decoding of the decoder transformer.
            This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            The model will output the same cache format that is fed as input.

            If `past_key_values` are used, the user can optionally input only the last `audio_values` or `audio_codes (those that don't
            have their past key value states given to this model).
    """

    audio_values: torch.FloatTensor = None
    semantic_tokens: torch.FloatTensor = None
    semantic_commitment_loss: torch.FloatTensor = None
    acoustic_commitment_loss: torch.FloatTensor = None


def default(val: tp.Any, d: tp.Any) -> tp.Any:
    return val if val is not None else d


def ema_inplace(moving_avg, new, decay: float):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def laplace_smoothing(x, n_categories: int, epsilon: float = 1e-5):
    return (x + epsilon) / (x.sum() + n_categories * epsilon)


def uniform_init(*shape: int):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


def sample_vectors(samples, num: int):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


def kmeans(samples, num_clusters: int, num_iters: int = 10):
    dim, dtype = samples.shape[-1], samples.dtype

    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        diffs = rearrange(samples, "n d -> n () d") - rearrange(
            means, "c d -> () c d"
        )
        dists = -(diffs ** 2).sum(dim=-1)

        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, "n -> n d", d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins

class MeomeoEuclideanCodebook(MimiEuclideanCodebook):
    """Codebook with Euclidean distance."""

    def __init__(self, config: MeomeoConfig, epsilon: float = 1e-5):
        super().__init__(config)
        self.accum_steps = config.gen_gradient_accumulation_steps
        self.accum_counter = 0
        self.codebook_decay = config.codebook_decay
        self.codebook_size = config.codebook_size
        self.codebook_dim = config.codebook_dim
        self.kmeans_init = config.kmeans_init
        self.kmeans_iters = config.kmeans_iters
        self.threshold_ema_dead_code = config.threshold_ema_dead_code
        init_fn: tp.Union[tp.Callable[..., torch.Tensor], tp.Any] = uniform_init if not self.kmeans_init else torch.zeros

        emb = init_fn(self.codebook_size, self.codebook_dim)

        # self.codebook_size = config.codebook_size

        self.register_buffer("inited", torch.Tensor([not self.kmeans_init]))
        self.register_buffer("cluster_size", torch.zeros(self.codebook_size))
        self.register_buffer("emb", emb)
        self.register_buffer("embed_avg", emb.clone())
        self.epsilon = epsilon

        self.accum_cluster_size = torch.zeros(self.codebook_size, device=self.emb.device).requires_grad_(False)
        self.accum_embed_avg = torch.zeros_like(self.emb, device=self.emb.device).requires_grad_(False)
        self.accum_samples = torch.empty(0, self.codebook_dim, device=self.emb.device).requires_grad_(False)

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.inited:
            # print('inited true')
            return

        emb, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)
        self.emb.data.copy_(emb)
        self.embed_avg.data.copy_(emb.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.inited.data.copy_(torch.Tensor([True]))
        # Make sure all buffers across workers are in sync after initialization
        #broadcast_tensors(self.buffers())

    def replace_(self, samples, mask):
        modified_codebook = torch.where(
            mask[..., None], sample_vectors(samples, self.codebook_size), self.emb
        )
        self.emb.data.copy_(modified_codebook)

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        # print('number of expired codes:', torch.sum(expired_codes).item())
        # print(expired_codes.numel())
        if not torch.any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, "... d -> (...) d")
        self.replace_(batch_samples, mask=expired_codes)
        #broadcast_tensors(self.buffers())

    # @property
    # def embed(self) -> torch.Tensor:
    #     if self._embed is None:
    #         self._embed = self.embed_sum / self.cluster_usage.clamp(min=self.epsilon)[:, None]
    #     return self._embed

    def quantize(self, x):
        emb = self.emb.t()
        dist = -(
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ emb
            + emb.pow(2).sum(0, keepdim=True)
        )
        embed_ind = dist.max(dim=-1).indices
        return embed_ind

    # Copied from transformers.models.encodec.modeling_encodec.EncodecEuclideanCodebook.encode
    def encode(self, hidden_states):
        shape = hidden_states.shape
        # pre-process
        hidden_states = hidden_states.reshape((-1, shape[-1]))
        # quantize
        embed_ind = self.quantize(hidden_states)
        # post-process
        embed_ind = embed_ind.view(*shape[:-1])
        return embed_ind

    # Copied from transformers.models.encodec.modeling_encodec.EncodecEuclideanCodebook.decode
    def decode(self, embed_ind):
        quantize = F.embedding(embed_ind, self.emb)
        return quantize

    def forward(self, x):
        shape, dtype = x.shape, x.dtype
        x = x.reshape((-1, shape[-1]))

        # print("Codebook Weights Loaded:", self.emb.mean().item())
        # print("Inited Status:", self.inited.item())

        self.init_embed_(x)

        embed_ind = self.quantize(x)
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = embed_ind.view(*shape[:-1])
        quantize = self.decode(embed_ind)

        if self.training:
            # print('counter', self.accum_counter)
            # print('accum step', self.accum_steps)
            # print(self.emb.device)
            # print(self.accum_cluster_size.device, embed_onehot.device)
            self.accum_cluster_size = self.accum_cluster_size.to(self.emb.device)
            self.accum_embed_avg = self.accum_embed_avg.to(self.emb.device)
            self.accum_samples = self.accum_samples.to(self.emb.device)

            self.accum_cluster_size += embed_onehot.sum(0).detach()
            self.accum_embed_avg += (x.t() @ embed_onehot).t().detach()
            # self.accum_samples.append(x.detach())
            self.accum_samples = torch.cat([self.accum_samples, x.detach()], dim=0)
            self.accum_counter += 1

            if self.accum_counter == self.accum_steps:
                # full_accumulated_samples = torch.cat(self.accum_samples, dim=0)
                # self.expire_codes_(full_accumulated_samples)
                self.expire_codes_(self.accum_samples)

                ema_inplace(self.cluster_size, self.accum_cluster_size, self.codebook_decay)
                ema_inplace(self.embed_avg, self.accum_embed_avg, self.codebook_decay)
                cluster_size = (
                    laplace_smoothing(self.cluster_size, self.codebook_size, self.epsilon)
                    * self.cluster_size.sum()
                )

                embed_normalized = self.embed_avg / (cluster_size.unsqueeze(1) + self.epsilon)
                self.emb.data.copy_(embed_normalized)

                # self.accum_cluster_size.zero_()
                # self.accum_embed_avg.zero_()
                # self.accum_samples = []
                # self.accum_counter = 0
                self.accum_cluster_size.zero_()
                self.accum_embed_avg.zero_()
                self.accum_samples = torch.empty(0, self.codebook_dim, device=self.emb.device)
                self.accum_counter = 0

        return quantize, embed_ind
    

class MeomeoVectorQuantization(MimiVectorQuantization):
    def __init__(self, config: MeomeoConfig):
        super().__init__(config)
        print('hello MeomeoVectorQuantization')
        self.codebook = MeomeoEuclideanCodebook(config)
        # self.commitment_loss_lambda = config.commitment_loss_lambda

    def encode(self, hidden_states):
        hidden_states = hidden_states.permute(0, 2, 1)
        embed_in = self.codebook.encode(hidden_states)
        return embed_in

    def decode(self, embed_ind):
        quantize = self.codebook.decode(embed_ind)
        quantize = quantize.permute(0, 2, 1)
        return quantize

    def forward(self, hidden_states):
        device = hidden_states.device
        hidden_states = hidden_states.permute(0, 2, 1)
        quantize, embed_ind = self.codebook(hidden_states)
        if self.training:
            quantize = hidden_states + (quantize - hidden_states).detach()
        loss = torch.tensor([0.0], device=device, requires_grad=self.training)
        if self.training:
            # if self.commitment_loss_lambda > 0:
            #     commit_loss = F.mse_loss(quantize.detach(), hidden_states, reduction="none")
            #     commit_loss = commit_loss.mean(dim=[1, 2])
            #     loss = loss + self.commitment_loss_lambda * commit_loss
            commit_loss = F.mse_loss(quantize.detach(), hidden_states, reduction="none")
            commit_loss = commit_loss.mean(dim=[1, 2])
            loss = loss + commit_loss


        quantize = quantize.permute(0, 2, 1)
        # print('quantize shape', quantize.shape)
        # print('embed_ind shape', embed_ind.shape)
        return quantize, embed_ind, loss


class MeomeoResidualVectorQuantizer(MimiResidualVectorQuantizer):
    """Residual Vector Quantizer."""

    def __init__(self, config: MeomeoConfig, max_num_quantizers: int, min_num_quantizers: int, quantization_skipping_rate: float):
        super().__init__(config, max_num_quantizers) #todo: more verify
        print('hello MeomeoResidualVectorQuantizer')

        self.codebook = config.codebook_size
        self.frame_rate = config.frame_rate
        # self.num_quantizers = num_quantizers if num_quantizers is not None else config.num_quantizers
        self.max_num_quantizers = max_num_quantizers
        self.min_num_quantizers = min_num_quantizers
        self.quantization_skipping_rate = quantization_skipping_rate

        self.layers = nn.ModuleList(
            [
                MeomeoVectorQuantization(config)
                for _ in range(self.max_num_quantizers)
            ]
        )
        self.input_proj = None
        self.output_proj = None
        if config.vector_quantization_hidden_dimension != config.hidden_size:
            self.input_proj = torch.nn.Conv1d(
                config.hidden_size, config.vector_quantization_hidden_dimension, 1, bias=False
            )
            self.output_proj = torch.nn.Conv1d(
                config.vector_quantization_hidden_dimension, config.hidden_size, 1, bias=False
            )

    def encode(self, embeddings: torch.Tensor, num_quantizers: Optional[int] = None) -> torch.Tensor:
        """
        Encode a given input tensor with the specified frame rate at the given number of quantizers / codebooks. The RVQ encode method sets
        the appropriate number of quantizers to use and returns indices for each quantizer.
        """
        if self.input_proj is not None:
            embeddings = self.input_proj(embeddings)

        num_quantizers = num_quantizers if num_quantizers is not None else self.num_quantizers

        residual = embeddings
        all_indices = []
        for layer in self.layers[:num_quantizers]:
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            residual = residual - quantized
            all_indices.append(indices)
        out_indices = torch.stack(all_indices)
        return out_indices

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode the given codes of shape [B, K, T] to the quantized representation."""
        quantized_out = torch.tensor(0.0, device=codes.device)
        codes = codes.transpose(0, 1)
        for i, indices in enumerate(codes):
            layer = self.layers[i]
            quantized = layer.decode(indices)
            quantized_out = quantized_out + quantized

        if self.output_proj is not None:
            quantized_out = self.output_proj(quantized_out)
        return quantized_out
    
    def forward(self, embeddings: torch.Tensor, nq: int):
        if self.input_proj is not None:
            embeddings = self.input_proj(embeddings)

        quantized_list = []
        commitment_loss_list = []
        embed_ind_list = []
        residual = embeddings

        nq_tmp = nq if nq != None else self.max_num_quantizers
        
        for layer in self.layers[:nq_tmp]:
            # print('count')
            quantized, embed_ind, commitment_loss = layer(residual)
            residual = residual - quantized
            quantized_list.append(quantized)
            embed_ind_list.append(embed_ind)
            commitment_loss_list.append(commitment_loss)

        quantized_stack = torch.stack(quantized_list, dim=0)
        embed_ind_stack = torch.stack(embed_ind_list, dim=0)
        commitment_loss_stack = torch.stack(commitment_loss_list, dim=0)
        batch_size = quantized_stack.shape[1]
        if nq != None or self.min_num_quantizers == None:
            if nq != None:
                assert nq <= self.max_num_quantizers, f'nq cant be bigger than {self.max_num_quantizers}'

            # print('rvq infer')
            # print(quantized_stack.shape)
            quantized_out = quantized_stack.sum(dim=0)
            commitment_loss_out = commitment_loss_stack.sum(dim=0)
            embed_ind_out = embed_ind_stack

        else:
            assert self.min_num_quantizers < self.max_num_quantizers, f'min_nq has to be smaller than {self.max_num_quantizers}'
            nq_values = torch.randint(low=self.min_num_quantizers, high=self.max_num_quantizers + 1, size=(batch_size,))
            num_zero_nqs = int(batch_size * self.quantization_skipping_rate)
            if batch_size == 1:
                num_zero_nqs = 1 if torch.rand(1).item() < self.quantization_skipping_rate else 0
            zero_nq_indices = torch.randperm(batch_size)[:num_zero_nqs]
            nq_values[zero_nq_indices] = 0

            quantizer_indices = torch.arange(self.max_num_quantizers).unsqueeze(1)
            mask = quantizer_indices < nq_values.unsqueeze(0)
            quantized_mask = mask.unsqueeze(-1).unsqueeze(-1).to(quantized_stack.device)
            commitment_loss_mask = mask.to(commitment_loss_stack.device)
            embed_ind_mask = mask.unsqueeze(-1).to(embed_ind_stack)

            # print('quantized_stack shape', quantized_stack.shape)
            # print('commitment_loss_stack shape', commitment_loss_stack.shape)
            # print('embed_ind_stack shape', embed_ind_stack.shape)
            # print('quantized_mask shape', quantized_mask.shape)
            # print('commitment_loss_mask shape', commitment_loss_mask.shape)
            # print('embed_ind_mask shape', embed_ind_mask.shape)

            quantized_out = quantized_stack * quantized_mask
            commitment_loss_out = commitment_loss_stack * commitment_loss_mask
            embed_ind_out = embed_ind_stack * embed_ind_mask

            quantized_out = quantized_out.sum(dim=0)
            # print('quantized_out', quantized_out)
            # print('quantized_out shape', quantized_out.shape)
            # print('zero_id', zero_nq_indices)
            # print('embeddings', embeddings)
            # print('embeddings shape', embeddings.shape)
            quantized_out[zero_nq_indices] = embeddings[zero_nq_indices]

            commitment_loss_out = commitment_loss_out.sum(dim=0)

        if self.output_proj is not None:
            quantized_out = self.output_proj(quantized_out)

        return quantized_out, embed_ind_out, commitment_loss_out.mean()


class MeomeoSplitResidualVectorQuantizer(MimiSplitResidualVectorQuantizer):
    def __init__(self, config: MeomeoConfig):
        super().__init__(config)
        print('hello MeomeoSplitResidualVectorQuantizer')
        self.max_num_quantizers = config.num_quantizers

        self.num_semantic_quantizers = config.num_semantic_quantizers
        self.num_acoustic_quantizers = config.num_quantizers - config.num_semantic_quantizers

        self.semantic_min_num_quantizers = config.semantic_min_nq if config.semantic_min_nq != self.num_semantic_quantizers else None
        self.acoustic_min_num_quantizers = config.acoustic_min_nq if config.acoustic_min_nq != self.num_acoustic_quantizers else None
        self.semantic_quantization_skipping_rate = config.semantic_quantization_skipping_rate
        self.acoustic_quantization_skipping_rate = config.acoustic_quantization_skipping_rate

        self.semantic_residual_vector_quantizer = MeomeoResidualVectorQuantizer(config, self.num_semantic_quantizers, self.semantic_min_num_quantizers, self.semantic_quantization_skipping_rate)
        self.acoustic_residual_vector_quantizer = MeomeoResidualVectorQuantizer(config, self.num_acoustic_quantizers, self.acoustic_min_num_quantizers, self.acoustic_quantization_skipping_rate)

    def encode(self, embeddings: torch.Tensor, num_quantizers: Optional[float] = None) -> torch.Tensor:
        """
        Encode a given input tensor with the specified frame rate at the given number of quantizers / codebooks. The RVQ encode method sets
        the appropriate number of quantizers to use and returns indices for each quantizer.
        """

        num_quantizers = self.max_num_quantizers if num_quantizers is None else num_quantizers

        if num_quantizers > self.max_num_quantizers:
            raise ValueError(
                f"The number of quantizers (i.e codebooks) asked should be lower than the total number of quantizers {self.max_num_quantizers}, but is currently {num_quantizers}."
            )

        if num_quantizers < self.num_semantic_quantizers:
            raise ValueError(
                f"The number of quantizers (i.e codebooks) asked should be higher than the number of semantic quantizers {self.num_semantic_quantizers}, but is currently {num_quantizers}."
            )

        # codes is [K, B, T], with T frames, K nb of codebooks.
        codes = self.semantic_residual_vector_quantizer.encode(embeddings)

        if num_quantizers > self.num_semantic_quantizers:
            acoustic_codes = self.acoustic_residual_vector_quantizer.encode(
                embeddings, num_quantizers=num_quantizers - self.num_semantic_quantizers
            )
            codes = torch.cat([codes, acoustic_codes], dim=0)

        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode the given codes to the quantized representation."""

        # The first num_semantic_quantizers codebooks are decoded using the semantic RVQ
        quantized_out = self.semantic_residual_vector_quantizer.decode(codes[:, : self.num_semantic_quantizers])

        # The rest of the codebooks are decoded using the acoustic RVQ
        if codes.shape[1] > self.num_semantic_quantizers:
            quantized_out += self.acoustic_residual_vector_quantizer.decode(codes[:, self.num_semantic_quantizers :])
        return quantized_out

    def forward(self, embeddings: torch.Tensor, semantic_nq: int, acoustic_nq: int):
        semantic_out, semantic_codes, semantic_commitment_loss = self.semantic_residual_vector_quantizer(embeddings, semantic_nq)
        acoustic_out, acoustic_codes, acoustic_commitment_loss = self.acoustic_residual_vector_quantizer(embeddings, acoustic_nq)
        quantized_out = semantic_out + acoustic_out
        codes = torch.cat([semantic_codes, acoustic_codes], dim=0)
        return quantized_out, semantic_out, codes, semantic_commitment_loss, acoustic_commitment_loss



class MeomeoModel(MimiModel):
    def __init__(self, config: MeomeoConfig):
        super().__init__(config)
        print('hello MeomeoModel')

        # Replace the quantizer with the updated version
        self.quantizer = MeomeoSplitResidualVectorQuantizer(config)

        # Add a projection layer for the semantic token
        self.semantic_token_projector = nn.Linear(self.config.hidden_prj_size, self.config.output_prj_size)


    def forward(
        self,
        input_values: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        semantic_nq: Optional[int] = None,
        acoustic_nq: Optional[int] = None,
        encoder_past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        decoder_past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        return_dict: Optional[bool] = None,
    ):
       
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if padding_mask is None:
            padding_mask = torch.ones_like(input_values).bool()

        _, channels, input_length = input_values.shape

        if channels < 1 or channels > 2:
            raise ValueError(f"Number of audio channels must be 1 or 2, but got {channels}")

        embeddings = self.encoder(input_values)
        encoder_outputs = self.encoder_transformer(
            embeddings.transpose(1, 2), past_key_values=encoder_past_key_values, return_dict=return_dict
        )
        if return_dict:
            encoder_past_key_values = encoder_outputs.get("past_key_values")
        elif len(encoder_outputs) > 1:
            encoder_past_key_values = encoder_outputs[1]
        embeddings = encoder_outputs[0].transpose(1, 2)
        embeddings = self.downsample(embeddings)

        embeddings, semantic_tokens, codes, semantic_commitment_loss, acoustic_commitment_loss = self.quantizer(embeddings, semantic_nq, acoustic_nq)
        codes = codes.transpose(0, 1)

        embeddings = self.upsample(embeddings)
        # semantic_tokens = self.upsample(semantic_tokens)
        semantic_tokens = semantic_tokens.transpose(1, 2)
        semantic_tokens = self.semantic_token_projector(semantic_tokens)
        decoder_outputs = self.decoder_transformer(
            embeddings.transpose(1, 2), past_key_values=decoder_past_key_values, return_dict=return_dict
        )
        if return_dict:
            decoder_past_key_values = decoder_outputs.get("past_key_values")
        elif len(decoder_outputs) > 1:
            decoder_past_key_values = decoder_outputs[1]
        
        embeddings = decoder_outputs[0].transpose(1, 2)
        audio_values = self.decoder(embeddings)

        if padding_mask is not None and padding_mask.shape[-1] < audio_values.shape[-1]:
            audio_values = audio_values[..., : padding_mask.shape[-1]]

        if not return_dict:
            return (audio_values, semantic_tokens, semantic_commitment_loss, acoustic_commitment_loss, encoder_past_key_values, decoder_past_key_values)


        return MeomeoOutput(
            audio_values=audio_values,
            semantic_tokens=semantic_tokens,
            audio_codes=codes,
            semantic_commitment_loss=semantic_commitment_loss,
            acoustic_commitment_loss=acoustic_commitment_loss,
            encoder_past_key_values=encoder_past_key_values,
            decoder_past_key_values=decoder_past_key_values,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # Get the config from pretrained model
        config = kwargs.pop("config", None)
        if config is None:
            config = MeomeoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Instantiate the model with the config
        model = cls(config, *model_args, **kwargs)

        # Check for state_dict
        state_dict = kwargs.pop("state_dict", None)
        if state_dict is None:
            try:
                resolved_weights_file = cached_file(pretrained_model_name_or_path, SAFE_WEIGHTS_NAME)
                state_dict = safetensors.torch.load_file(resolved_weights_file, device="cpu")
            except (OSError, safetensors.torch.SafeTensorError):
                resolved_weights_file = cached_file(pretrained_model_name_or_path, WEIGHTS_NAME)
                state_dict = torch.load(resolved_weights_file, map_location="cpu")

        # for key, value in state_dict.items():
        #     if "cluster_usage" in key:
        #         print(value)
        #         print(type(value))
        #         mask = value < 0.2
        #         print(mask.numel())
        #         print(torch.sum(mask).item())
        # Convert pretrained codebook keys to Meomeo format
        updated_state_dict = {}
        for key, value in state_dict.items():
            if "codebook" in key:
                new_key = key.replace("cluster_usage", "cluster_size")
                new_key = new_key.replace("embed_sum", "emb")
                new_key = new_key.replace("initialized", "inited")
                
                # Compute embed_avg from embed_sum and cluster_usage if needed
                if "emb" in new_key:
                    cluster_key = key.replace("embed_sum", "cluster_usage")
                    if cluster_key in state_dict:
                        cluster_usage = state_dict[cluster_key].clamp(min=1e-5)
                        value = value / cluster_usage[:, None]  # Normalize emb
                    else:
                        print(f"Warning: Missing {cluster_key} when computing embed_avg")
                
                updated_state_dict[new_key] = value
            else:
                updated_state_dict[key] = value

        # Load the updated state_dict
        missing_keys, unexpected_keys = model.load_state_dict(updated_state_dict, strict=False)
        # print("Missing keys:", missing_keys)
        # print("Unexpected keys:", unexpected_keys)

        return model
