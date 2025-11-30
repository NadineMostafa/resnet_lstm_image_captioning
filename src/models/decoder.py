import random
import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    A decoder for generating captions from image features using an LSTM.
    """

    def __init__(self, embed_size, vocab_size, hidden_size, num_layers=1):
        """
        Initialize the decoder.

        Args:
            embed_size (int): Size of the word embedding vector.
            vocab_size (int): Size of the vocabulary.
            hidden_size (int): Number of features in the LSTM hidden state.
            num_layers (int, optional): Number of LSTM layers. Defaults to 1.
        """
        super().__init__()
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions, teacher_forcing_ratio=1.0):
        """
        Perform a forward pass through the decoder.

        Args:
            features (Tensor): Image features of shape (B, embed_size).
            captions (Tensor): Ground truth captions of shape (B, seq_len).
            teacher_forcing_ratio (float, optional): Probability of using ground truth tokens 
                                                     during training. Defaults to 1.0.

        Returns:
            Tensor: Predicted word scores of shape (B, seq_len-1, vocab_size).
        """
        seq_len = captions.size(1)
        hidden_state = None
        input_token = features.unsqueeze(1)  # (B, 1, embed_size)
        outputs = []

        for t in range(seq_len - 1):
            # LSTM step
            lstm_out, hidden_state = self.lstm(input_token, hidden_state)
            output = self.linear(lstm_out)  # (B, 1, vocab_size)
            outputs.append(output)

            if self.training and teacher_forcing_ratio < 1.0:
                use_teacher_forcing = random.random() < teacher_forcing_ratio
                if use_teacher_forcing:
                    input_token = self.embedding(captions[:, t + 1].unsqueeze(1))
                else:
                    predicted_token = output.argmax(dim=-1)
                    input_token = self.embedding(predicted_token)
            else:
                input_token = self.embedding(captions[:, t + 1].unsqueeze(1))

        return torch.cat(outputs, dim=1)  # (B, seq_len-1, vocab_size)