from collections import Counter
from .tokenizer import Tokenizer

class Vocab:
    def __init__(self):
        self.word_freq = Counter()
        self.all_tokens = list()
        self.tokenizer = Tokenizer()
        self.word2idx = {
            "<PAD>": 0,
            "<SOS>": 1,
            "<EOS>": 2,
            "<UNK>": 3
        }
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def build_vocab(self, captions):
        for caption in captions:
            for sentence in caption:
                tokens = self.tokenizer.tokenize(sentence.lower())
                self.all_tokens.extend(tokens)
    
        self.word_freq = Counter(self.all_tokens)

    def build_mappings(self, min_freq=5):
        idx = 4
        for word, freq in self.word_freq.items():
            if freq >= min_freq:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

    def __len__(self):
        return len(self.word2idx)
    
    def word_to_index(self, word):
        return self.word2idx.get(word, self.word2idx["<UNK>"])
