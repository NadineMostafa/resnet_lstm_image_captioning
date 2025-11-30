from collections import Counter
from .tokenizer import Tokenizer

class Vocab:
    """
    A vocabulary class for building word-to-index and index-to-word mappings.
    """

    def __init__(self):
        """
        Initialize the vocabulary with special tokens and a tokenizer.
        """
        self.word_freq = Counter()
        self.all_tokens = list()
        self.tokenizer = Tokenizer()
        self.word2idx = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<SOS>": 2,
            "<EOS>": 3
        }
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def build_vocab(self, captions):
        """
        Build the vocabulary by counting word frequencies from captions.

        Args:
            captions (list): A list of caption lists to process.
        """
        for caption in captions:
            for sentence in caption:
                tokens = self.tokenizer.tokenize(sentence.lower())
                self.all_tokens.extend(tokens)
    
        self.word_freq = Counter(self.all_tokens)

    def build_mappings(self, min_freq=5):
        """
        Create word-to-index and index-to-word mappings based on word frequency.

        Args:
            min_freq (int): Minimum frequency for a word to be included in the vocabulary.
        """
        idx = 4
        for word in sorted(self.word_freq.keys()):
            freq = self.word_freq[word]
            if freq >= min_freq:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

    def __len__(self):
        return len(self.word2idx)
    
    def word_to_index(self, word):
        return self.word2idx.get(word, self.word2idx["<UNK>"])
