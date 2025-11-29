import torch.nn as nn
from .encoder import CNNEncoder
from .decoder import Decoder

class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, vocab_size, hidden_size):
        super().__init__()
        self.encoder = CNNEncoder(embed_size)
        self.decoder = Decoder(embed_size, vocab_size, hidden_size, num_layers=1)

    def forward(self, images, captions):
        features = self.encoder(images)
        output = self.decoder(features, captions)
        return output