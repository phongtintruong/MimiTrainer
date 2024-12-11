from typing import Optional, Union, List, Tuple
from dataclasses import dataclass
import torch
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers import MimiModel
from transformers.models.mimi.modeling_mimi import MimiSplitResidualVectorQuantizer, MimiOutput, MimiDecoderOutput, MimiModel

@dataclass
class TrainingMimiOutput(MimiOutput):
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

    audio_codes: torch.LongTensor = None
    audio_values: torch.FloatTensor = None
    semantic_token: torch.FloatTensor = None
    encoder_past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None
    decoder_past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None


@dataclass
class TrainingMimiDecoderOutput(MimiDecoderOutput):
    """
    Args:
        audio_values (`torch.FloatTensor`  of shape `(batch_size, segment_length)`, *optional*):
            Decoded audio values, obtained using the decoder part of Mimi.
        decoder_past_key_values (`Cache`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks) that can be used to speed up sequential decoding of the decoder transformer.
            This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            The model will output the same cache format that is fed as input.

            If `past_key_values` are used, the user can optionally input only the last `audio_values` or `audio_codes (those that don't
            have their past key value states given to this model).
    """

    audio_values: torch.FloatTensor = None
    semantic_token: torch.FloatTensor = None
    decoder_past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None


class TrainingMimiSplitResidualVectorQuantizer(MimiSplitResidualVectorQuantizer):
    def decode(self, codes: torch.Tensor) -> dict:
        """
                Decode the given codes to the quantized representation.

                Returns a dictionary with:
                - `quantized_semantic`: The quantized output from the semantic residual vector quantizer.
                - `quantized_acoustic`: The quantized output from the acoustic residual vector quantizer.
                - `quantized_combined`: The final combined quantized output.
                """

        # Decode semantic quantizers
        quantized_semantic = self.semantic_residual_vector_quantizer.decode(codes[:, : self.num_semantic_quantizers])

        # Decode acoustic quantizers, if present
        quantized_acoustic = None
        if codes.shape[1] > self.num_semantic_quantizers:
            quantized_acoustic = self.acoustic_residual_vector_quantizer.decode(codes[:, self.num_semantic_quantizers:])
            quantized_combined = quantized_semantic + quantized_acoustic
        else:
            quantized_combined = quantized_semantic

        return {
            "quantized_semantic": quantized_semantic,
            "quantized_acoustic": quantized_acoustic,
            "quantized_combined": quantized_combined,
        }


class TrainingMimiModel(MimiModel):
    def __init__(self, config):
        super().__init__(config)
        print('hello TrainingMimiModel')

        # Replace the quantizer with the updated version
        self.quantizer = TrainingMimiSplitResidualVectorQuantizer(config)

    def _decode_frame(
        self,
        codes: torch.Tensor,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Union[Cache, List[torch.FloatTensor]]]]:
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        quantization_outputs = self.quantizer.decode(codes)
        embeddings = quantization_outputs['quantized_combined']
        semantic_token = quantization_outputs['quantized_semantic']

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
        return outputs, semantic_token, past_key_values

    def decode(
        self,
        audio_codes: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        decoder_past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], TrainingMimiDecoderOutput]:
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

        audio_values, semantic_token, decoder_past_key_values = self._decode_frame(
            audio_codes, past_key_values=decoder_past_key_values, return_dict=return_dict
        )

        # truncate based on padding mask
        if padding_mask is not None and padding_mask.shape[-1] < audio_values.shape[-1]:
            audio_values = audio_values[..., : padding_mask.shape[-1]]

        if not return_dict:
            return (
                audio_values,
                semantic_token,
                decoder_past_key_values,
            )
        return TrainingMimiDecoderOutput(audio_values, semantic_token, decoder_past_key_values)

    def forward(
        self,
        input_values: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        num_quantizers: Optional[int] = None,
        audio_codes: Optional[torch.Tensor] = None,
        encoder_past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        decoder_past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], TrainingMimiOutput]:
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
                input_values, padding_mask, num_quantizers, encoder_past_key_values, return_dict=return_dict
            )
            audio_codes = encoder_outputs[0]
            if return_dict:
                encoder_past_key_values = encoder_outputs.get("past_key_values")
            elif len(encoder_outputs) > 1:
                encoder_past_key_values = encoder_outputs[1]

        decoder_outputs = self.decode(audio_codes, padding_mask, decoder_past_key_values, return_dict=return_dict)

        if return_dict:
            audio_values = decoder_outputs.audio_values
            semantic_token = decoder_outputs.semantic_token
            decoder_past_key_values = decoder_outputs.decoder_past_key_values
        else:
            audio_values, semantic_token, decoder_past_key_values = decoder_outputs

        print(semantic_token)

        if not return_dict:
            return (audio_codes, audio_values, semantic_token, encoder_past_key_values, decoder_past_key_values)

        return TrainingMimiOutput(
            audio_codes=audio_codes,
            audio_values=audio_values,
            semantic_token=semantic_token,
            encoder_past_key_values=encoder_past_key_values,
            decoder_past_key_values=decoder_past_key_values,
        )