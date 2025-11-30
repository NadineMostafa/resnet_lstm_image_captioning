import torch.nn as nn
from .encoder import CNNEncoder
from .decoder import Decoder

class ImageCaptioningModel(nn.Module):
    """
    An image captioning model combining a CNN encoder and an LSTM decoder.
    """

    def __init__(self, embed_size, vocab_size, hidden_size):
        """
        Initialize the image captioning model.

        Args:
            embed_size (int): Size of the embedding vector for image features and words.
            vocab_size (int): Size of the vocabulary.
            hidden_size (int): Number of features in the LSTM hidden state.
        """
        super().__init__()
        self.encoder = CNNEncoder(embed_size)
        self.decoder = Decoder(embed_size, vocab_size, hidden_size, num_layers=1)

    def forward(self, images, captions):
        """
        Forward pass through the model.

        Args:
            images (Tensor): Input images of shape (B, C, H, W).
            captions (Tensor): Ground truth captions of shape (B, seq_len).

        Returns:
            Tensor: Predicted word scores of shape (B, seq_len-1, vocab_size).
        """
        features = self.encoder(images)
        output = self.decoder(features, captions)
        return output