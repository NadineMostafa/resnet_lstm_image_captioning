import nltk
from nltk.tokenize import word_tokenize

class Tokenizer():
    """
    A simple tokenizer for text using NLTK's word tokenizer.
    """

    def __init__(self):
        """
        Initialize the tokenizer and download necessary NLTK resources.
        """
        nltk.download('punkt')
        nltk.download('punkt_tab')

    def tokenize(self, text):
        """
        Tokenize the input text into words.

        Args:
            text (str): The input text to tokenize.

        Returns:
            list: A list of tokenized words.
        """
        return word_tokenize(text)
