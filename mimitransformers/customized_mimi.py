from typing import Optional, Union, List, Tuple
import typing as tp
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers import MimiModel, MimiConfig
from transformers.models.mimi.modeling_mimi import MimiSplitResidualVectorQuantizer, MimiOutput, MimiEncoderOutput, MimiModel, MimiResidualVectorQuantizer, MimiDecoderOutput, MimiEuclideanCodebook, MimiVectorQuantization
from torch import nn
from transformers import PreTrainedModel
from transformers.utils import cached_file, WEIGHTS_NAME, SAFE_WEIGHTS_NAME
import safetensors.torch
from einops import rearrange, repeat


class MeomeoConfig(MimiConfig):

    model_type = "meomeo"

    def __init__(self, hidden_prj_size=512, output_prj_size=1024, **kwargs):
        super().__init__(**kwargs)
        self.hidden_prj_size = hidden_prj_size
        self.output_prj_size = output_prj_size

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

    semantic_tokens: torch.FloatTensor = None
    audio_values: torch.FloatTensor = None


@dataclass
class MeomeoEncoderOutput(MimiEncoderOutput):
    """
    Args:
        audio_codes (`torch.LongTensor`  of shape `(batch_size, num_quantizers, codes_length)`, *optional*):
            Discret code embeddings computed using `model.encode`.
        encoder_past_key_values (`Cache`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks) that can be used to speed up sequential decoding of the encoder transformer.
            This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            The model will output the same cache format that is fed as input.

            If `past_key_values` are used, the user can optionally input only the last `audio_values` or `audio_codes (those that don't
            have their past key value states given to this model).
    """

    audio_codes: torch.LongTensor = None
    semantic_tokens: torch.FloatTensor = None
    encoder_past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None


@dataclass
class MeomeoNoQuantizationEncoderOutput(MimiEncoderOutput):
    """
    Args:
        audio_codes (`torch.LongTensor`  of shape `(batch_size, num_quantizers, codes_length)`, *optional*):
            Discret code embeddings computed using `model.encode`.
        encoder_past_key_values (`Cache`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks) that can be used to speed up sequential decoding of the encoder transformer.
            This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            The model will output the same cache format that is fed as input.

            If `past_key_values` are used, the user can optionally input only the last `audio_values` or `audio_codes (those that don't
            have their past key value states given to this model).
    """

    embeddings: torch.FloatTensor = None
    semantic_tokens: torch.FloatTensor = None
    encoder_past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None

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
        super().__init__()
        self.codebook_decay = config.codebook_decay
        self.codebook_size = config.codebook_size
        self.codebook_dim = config.codebook_dim
        self.kmeans_init = config.kmeans_init
        self.kmeans_iters = config.kmeans_iters
        self.threshold_ema_dead_code = config.threshold_ema_dead_code
        init_fn: tp.Union[tp.Callable[..., torch.Tensor], tp.Any] = uniform_init if not self.kmeans_init else torch.zeros

        embed = init_fn(self.codebook_size, self.codebook_dim)

        # self.codebook_size = config.codebook_size

        self.register_buffer("inited", torch.Tensor([not self.kmeans_init]))
        self.register_buffer("cluster_size", torch.zeros(self.codebook_size))
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())
        self.epsilon = epsilon

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.inited:
            return

        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.inited.data.copy_(torch.Tensor([True]))
        # Make sure all buffers across workers are in sync after initialization
        #broadcast_tensors(self.buffers())

    def replace_(self, samples, mask):
        modified_codebook = torch.where(
            mask[..., None], sample_vectors(samples, self.codebook_size), self.embed
        )
        self.embed.data.copy_(modified_codebook)

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code
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
        embed = self.embed.t()
        dist = -(
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ embed
            + embed.pow(2).sum(0, keepdim=True)
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
        quantize = F.embedding(embed_ind, self.embed)
        return quantize

    def forward(self, x):
        shape, dtype = x.shape, x.dtype
        x = x.reshape((-1, shape[-1]))

        self.init_embed_(x)

        embed_ind = self.quantize(x)
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = embed_ind.view(*shape[:-1])
        quantize = self.decode(embed_ind)

        if self.training:
            self.expire_codes_(x)
            ema_inplace(self.cluster_size, embed_onehot.sum(0), self.codebook_decay)
            embed_sum = x.t() @ embed_onehot
            ema_inplace(self.embed_avg, embed_sum.t(), self.codebook_decay)
            cluster_size = (
                laplace_smoothing(self.cluster_size, self.codebook_size, self.epsilon)
                * self.cluster_size.sum()
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)

        return quantize, embed_ind
    

class MeomeoVectorQuantization(MimiVectorQuantization):
    def __init__(self, config: MeomeoConfig):
        super().__init__(config)
        print('hello MeomeoVectorQuantization')
        self.codebook = MeomeoEuclideanCodebook(config)
        self.commitment_loss_lambda = config.commitment_loss_lambda

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
            if self.commitment_loss_lambda > 0:
                commit_loss = F.mse_loss(quantize.detach(), hidden_states, reduction="none")
                commit_loss = commit_loss.mean(dim=[1, 2])
                loss = loss + self.commitment_loss_lambda * commit_loss

        quantize = quantize.permute(0, 2, 1)
        return quantize, embed_ind, loss


class MeomeoResidualVectorQuantizer(MimiResidualVectorQuantizer):
    """Residual Vector Quantizer."""

    def __init__(self, config: MeomeoConfig, max_num_quantizers: int, min_num_quantizers: int, quantization_rate: float):
        super().__init__(config, max_num_quantizers) #todo: more verify
        print('hello MeomeoResidualVectorQuantizer')

        self.codebook = config.codebook_size
        self.frame_rate = config.frame_rate
        # self.num_quantizers = num_quantizers if num_quantizers is not None else config.num_quantizers
        self.max_num_quantizers = max_num_quantizers
        self.min_num_quantizers = min_num_quantizers
        self.quantization_rate = quantization_rate

        self.layers = nn.ModuleList(
            [
                MeomeoVectorQuantization(config)
                for _ in range(self.num_quantizers)
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
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.input_proj is not None:
            embeddings = self.input_proj(embeddings)

        quantized_list = []
        residual = embeddings
        
        for layer in self.layers[:self.max_num_quantizers]:
            quantized, embed_ind, commitment_loss = layer(residual)
            residual = residual - quantized
            quantized_list.append(quantized)

        quantized_stack = torch.stack(quantized_list, dim=0)

        
        

    # def encode(self, embeddings: torch.Tensor, num_quantizers: Optional[int] = None, output_quantized: Optional[bool] = False) -> torch.Tensor:
    #     """
    #     Encode a given input tensor with the specified frame rate at the given number of quantizers / codebooks. The RVQ encode method sets
    #     the appropriate number of quantizers to use and returns indices for each quantizer.
    #     """
    #     if self.input_proj is not None:
    #         embeddings = self.input_proj(embeddings)

    #     num_quantizers = num_quantizers if num_quantizers is not None else self.num_quantizers

    #     residual = embeddings.clone()
    #     final_quantized = None
    #     all_indices = []
    #     for layer in self.layers[:num_quantizers]:
    #         indices = layer.encode(residual)
    #         quantized = layer.decode(indices)
    #         final_quantized = quantized if final_quantized is None else final_quantized + quantized
    #         residual = residual - quantized
    #         all_indices.append(indices)
    #     out_indices = torch.stack(all_indices)
    #     if output_quantized:
    #         # print('final_quantized', final_quantized.shape)
    #         # print('original_embeddings', embeddings.shape)
    #         final_quantized = self.output_proj(final_quantized)
    #         # print('final_quantized', final_quantized.shape)
    #         return out_indices, final_quantized
    #     else:
    #         return out_indices



class MeomeoSplitResidualVectorQuantizer(MimiSplitResidualVectorQuantizer):
    def __init__(self, config: MeomeoConfig):
        super().__init__(config)
        print('hello MeomeoSplitResidualVectorQuantizer')

        # Replace the quantizer with the updated version
        self.semantic_residual_vector_quantizer = MeomeoResidualVectorQuantizer(config, self.num_semantic_quantizers)
        self.acoustic_residual_vector_quantizer = MeomeoResidualVectorQuantizer(config, self.num_acoustic_quantizers)
    def encode(self, embeddings: torch.Tensor, num_quantizers: Optional[float] = None, do_acoustic: Optional[bool] = True) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode a given input tensor with the specified frame rate at the given number of quantizers / codebooks. The RVQ encode method sets
        the appropriate number of quantizers to use and returns indices for each quantizer.
        """
        
        # codes is [K, B, T], with T frames, K nb of codebooks.
        codes, quantized = self.semantic_residual_vector_quantizer.encode(embeddings, output_quantized=True)

        if do_acoustic:
            num_quantizers = self.max_num_quantizers if num_quantizers is None else num_quantizers

            if num_quantizers > self.max_num_quantizers:
                raise ValueError(
                    f"The number of quantizers (i.e codebooks) asked should be lower than the total number of quantizers {self.max_num_quantizers}, but is currently {num_quantizers}."
                )

            if num_quantizers < self.num_semantic_quantizers:
                raise ValueError(
                    f"The number of quantizers (i.e codebooks) asked should be higher than the number of semantic quantizers {self.num_semantic_quantizers}, but is currently {num_quantizers}."
                )


            if num_quantizers > self.num_semantic_quantizers:
                acoustic_codes = self.acoustic_residual_vector_quantizer.encode(
                    embeddings, num_quantizers=num_quantizers - self.num_semantic_quantizers
                )
                codes = torch.cat([codes, acoustic_codes], dim=0)

            return codes, quantized
        else:
            return quantized



class MeomeoModel(MimiModel):
    def __init__(self, config: MeomeoConfig):
        super().__init__(config)
        print('hello MeomeoModel')

        # Replace the quantizer with the updated version
        self.quantizer = MeomeoSplitResidualVectorQuantizer(config)

        # Add a projection layer for the semantic token
        self.semantic_token_projector = nn.Linear(self.config.hidden_prj_size, self.config.output_prj_size)

    def _encode_frame(
        self,
        input_values: torch.Tensor,
        num_quantizers: int,
        padding_mask: int,
        do_quantize: bool,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Encodes the given input using the underlying VQVAE. The padding mask is required to compute the correct scale.
        """
        embeddings = self.encoder(input_values)
        encoder_outputs = self.encoder_transformer(
            embeddings.transpose(1, 2), past_key_values=past_key_values, return_dict=return_dict
        )
        if return_dict:
            past_key_values = encoder_outputs.get("past_key_values")
        elif len(encoder_outputs) > 1:
            past_key_values = encoder_outputs[1]
        embeddings = encoder_outputs[0].transpose(1, 2)
        embeddings = self.downsample(embeddings)

        if do_quantize:
            codes, semantic_token = self.quantizer.encode(embeddings, num_quantizers, do_acoustic=do_quantize)
            codes = codes.transpose(0, 1)
            # print('semantic_token in meomeo encode frame', semantic_token.shape)
            return codes, semantic_token, past_key_values
        else:
            semantic_token = self.quantizer.encode(embeddings, num_quantizers, do_acoustic=do_quantize)
            # print('semantic_token in meomeo encode frame no quantize', semantic_token.shape)
            return embeddings, semantic_token, past_key_values
    
    def encode(
        self,
        input_values: torch.Tensor,
        do_quantize: bool,
        padding_mask: torch.Tensor = None,
        num_quantizers: Optional[float] = None,
        encoder_past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], MeomeoEncoderOutput, MeomeoNoQuantizationEncoderOutput]:
        """
        Encodes the input audio waveform into discrete codes.

        Args:
            input_values (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`):
                Float values of the input audio waveform.
            padding_mask (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`):
                Indicates which inputs are to be ignored due to padding, where elements are either 1 for *not masked* or 0
                for *masked*.
            num_quantizers (`int`, *optional*):
                Number of quantizers (i.e codebooks) to use. By default, all quantizers are used.
            encoder_past_key_values (`Cache`, *optional*):
                Pre-computed hidden-states (key and values in the self-attention blocks) that can be used to speed up sequential decoding of the encoder transformer.
                This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

                The model will output the same cache format that is fed as input.

                If `past_key_values` are used, the user can optionally input only the last `audio_values` or `audio_codes (those that don't
                have their past key value states given to this model).
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:
            `codebook` of shape `[batch_size, num_codebooks, frames]`, the discrete encoded codes for the input audio waveform.
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        num_quantizers = self.config.num_quantizers if num_quantizers is None else num_quantizers

        if num_quantizers > self.config.num_quantizers:
            raise ValueError(
                f"The number of quantizers (i.e codebooks) asked should be lower than the total number of quantizers {self.config.num_quantizers}, but is currently {num_quantizers}."
            )

        _, channels, input_length = input_values.shape

        if channels < 1 or channels > 2:
            raise ValueError(f"Number of audio channels must be 1 or 2, but got {channels}")

        if padding_mask is None:
            padding_mask = torch.ones_like(input_values).bool()

        if do_quantize:
            encoded_frames, semantic_tokens, encoder_past_key_values = self._encode_frame(
                input_values,
                num_quantizers,
                padding_mask.bool(),
                do_quantize,
                past_key_values=encoder_past_key_values,
                return_dict=return_dict,
            )
            # print('semantic_tokens in meomeo encode', semantic_tokens.shape)

            if not return_dict:
                return (
                    encoded_frames,
                    semantic_tokens,
                    encoder_past_key_values,
                )

            return MeomeoEncoderOutput(audio_codes=encoded_frames, semantic_tokens=semantic_tokens, encoder_past_key_values=encoder_past_key_values)
        else:
            embeddings, semantic_tokens, encoder_past_key_values = self._encode_frame(
                input_values,
                num_quantizers,
                padding_mask.bool(),
                do_quantize,
                past_key_values=encoder_past_key_values,
                return_dict=return_dict,
            )
            # print('semantic_tokens in meomeo encode no quantize', semantic_tokens.shape)

            if not return_dict:
                return (
                    embeddings,
                    semantic_tokens,
                    encoder_past_key_values,
                )
            
            return MeomeoNoQuantizationEncoderOutput(embeddings=embeddings, semantic_tokens=semantic_tokens, encoder_past_key_values=encoder_past_key_values)

    def _decode_frame(
        self,
        codes: torch.Tensor,
        do_quantize: bool,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.Tensor:
        
        if do_quantize:
            embeddings = self.quantizer.decode(codes)
        else:
            embeddings = codes
        
        embeddings = self.upsample(embeddings)
        decoder_outputs = self.decoder_transformer(
            embeddings.transpose(1, 2), past_key_values=past_key_values, return_dict=return_dict
        )
        if return_dict:
            past_key_values = decoder_outputs.get("past_key_values")
        elif len(decoder_outputs) > 1:
            past_key_values = decoder_outputs[1]
        embeddings = decoder_outputs[0].transpose(1, 2)
        outputs = self.decoder(embeddings)
        return outputs, past_key_values
    
    def decode(
        self,
        audio_codes: torch.Tensor,
        do_quantize: bool,
        padding_mask: Optional[torch.Tensor] = None,
        decoder_past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], MimiDecoderOutput]:
        """
        Decodes the given frames into an output audio waveform.

        Note that the output might be a bit bigger than the input. In that case, any extra steps at the end can be
        trimmed.

        Args:
            audio_codes (`torch.LongTensor`  of shape `(batch_size, num_quantizers, codes_length)`, *optional*):
                Discret code embeddings computed using `model.encode`.
            padding_mask (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`):
                Indicates which inputs are to be ignored due to padding, where elements are either 1 for *not masked* or 0
                for *masked*.
            decoder_past_key_values (`Cache`, *optional*):
                Pre-computed hidden-states (key and values in the self-attention blocks) that can be used to speed up sequential decoding of the decoder transformer.
                This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

                The model will output the same cache format that is fed as input.

                If `past_key_values` are used, the user can optionally input only the last `audio_values` or `audio_codes (those that don't
                have their past key value states given to this model).
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        audio_values, decoder_past_key_values = self._decode_frame(
            audio_codes, do_quantize=do_quantize, past_key_values=decoder_past_key_values, return_dict=return_dict
        )

        # truncate based on padding mask
        if padding_mask is not None and padding_mask.shape[-1] < audio_values.shape[-1]:
            audio_values = audio_values[..., : padding_mask.shape[-1]]

        if not return_dict:
            return (
                audio_values,
                decoder_past_key_values,
            )
        return MimiDecoderOutput(audio_values, decoder_past_key_values)


    def forward(
        self,
        input_values: torch.Tensor,
        do_quantize: bool,
        padding_mask: Optional[torch.Tensor] = None,
        num_quantizers: Optional[int] = None,
        audio_codes: Optional[torch.Tensor] = None,
        encoder_past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        decoder_past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], MeomeoOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from datasets import load_dataset
        >>> from transformers import AutoFeatureExtractor, MimiModel

        >>> dataset = load_dataset("hf-internal-testing/ashraq-esc50-1-dog-example")
        >>> audio_sample = dataset["train"]["audio"][0]["array"]

        >>> model_id = "kyutai/mimi"
        >>> model = MimiModel.from_pretrained(model_id)
        >>> feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

        >>> inputs = feature_extractor(raw_audio=audio_sample, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> audio_codes = outputs.audio_codes
        >>> audio_values = outputs.audio_values
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if padding_mask is None:
            padding_mask = torch.ones_like(input_values).bool()

        if audio_codes is None:
            encoder_outputs = self.encode(
                input_values, do_quantize, padding_mask, num_quantizers, encoder_past_key_values, return_dict=return_dict
            )
            # print('encoder_outputs', encoder_outputs)
            if return_dict:
                if do_quantize:
                    audio_codes = encoder_outputs.audio_codes
                    semantic_tokens = encoder_outputs.semantic_tokens
                    encoder_past_key_values = encoder_outputs.encoder_past_key_values
                else:
                    embeddings = encoder_outputs.embeddings
                    semantic_tokens = encoder_outputs.semantic_tokens
                    encoder_past_key_values = encoder_outputs.encoder_past_key_values
            else:
                if do_quantize:
                    audio_codes = encoder_outputs[0]
                    semantic_tokens = encoder_outputs[1]
                    encoder_past_key_values = encoder_outputs[2]
                else:
                    embeddings = encoder_outputs[0]
                    semantic_tokens = encoder_outputs[1]
                    encoder_past_key_values = encoder_outputs[2]

        if do_quantize:
            decoder_outputs = self.decode(audio_codes, do_quantize, padding_mask, decoder_past_key_values, return_dict=return_dict)
        else:
            decoder_outputs = self.decode(embeddings, do_quantize, padding_mask, decoder_past_key_values, return_dict=return_dict)
        audio_values = decoder_outputs[0]
        if return_dict:
            decoder_past_key_values = decoder_outputs.get("past_key_values")
        elif len(decoder_outputs) > 1:
            decoder_past_key_values = decoder_outputs[1]

        semantic_tokens = semantic_tokens.transpose(1, 2)
        semantic_tokens = self.semantic_token_projector(semantic_tokens)
        # print('semantic_token', semantic_tokens.shape)
        # print('audio_values', audio_values.shape)

        if not return_dict:
            return (audio_values, semantic_tokens)

        return MeomeoOutput(
            audio_values=audio_values,
            semantic_tokens=semantic_tokens
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

        # If no state_dict is provided, attempt to resolve it
        if state_dict is None:
            try:
                # First, try to load safetensors weights
                resolved_weights_file = cached_file(pretrained_model_name_or_path, SAFE_WEIGHTS_NAME)
                state_dict = safetensors.torch.load_file(resolved_weights_file, device="cpu")
            except (OSError, safetensors.torch.SafeTensorError):
                # Fall back to pytorch_model.bin if safetensors are not available
                resolved_weights_file = cached_file(pretrained_model_name_or_path, WEIGHTS_NAME)
                state_dict = torch.load(resolved_weights_file, map_location="cpu")

        # Load weights (adapt for any custom layers, if needed)
        model.load_state_dict(state_dict, strict=False)

        return model
    