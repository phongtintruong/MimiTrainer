from transformers import Wav2Vec2Model, HubertModel
import torch.nn as nn

class SemanticTeacher(nn.Module):
    """
    A unified class to load and use HuBERT or Wav2Vec2 models,
    behaving like an instance of these models.
    """

    def __init__(self, model_name: str, model_type: str = "hubert"):
        """
        Initializes the speech model.

        Args:
            model_name (str): The Hugging Face model identifier (e.g., "facebook/hubert-large-ls960-ft").
            model_type (str): Model type, either "hubert" or "wav2vec2".
        """
        super().__init__()  # Initialize nn.Module

        if model_type.lower() == "hubert":
            self.model = HubertModel.from_pretrained(model_name)
        elif model_type.lower() == "wav2vec2":
            self.model = Wav2Vec2Model.from_pretrained(model_name)
        else:
            raise ValueError("Invalid model_type. Choose 'hubert' or 'wav2vec2'.")

        self.config = self.model.config  # Store model config
        self.model.eval()  # Set model to evaluation mode

    @classmethod
    def from_pretrained(cls, model_name: str, model_type: str = "hubert"):
        """Alternative way to load a pretrained model."""
        return cls(model_name, model_type)

    def forward(self, input_values, output_hidden_states=False):
        """
        Forward pass to extract hidden states.

        Args:
            input_values (torch.Tensor): The input waveform tensor (batch_size, sequence_length).

        Returns:
            torch.Tensor: Last hidden state (batch_size, seq_len, hidden_dim).
        """
        return self.model(input_values, output_hidden_states=output_hidden_states)  # Mimic model behavior
