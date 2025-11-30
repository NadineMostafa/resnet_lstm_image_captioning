from torch.utils.data import Dataset

class ImageCaptioningDataset(Dataset):
    """
    A PyTorch Dataset for image-captioning tasks using a Hugging Face dataset.
    Each item corresponds to an image-caption pair, with captions tokenized and indexed.
    """

    def __init__(self, hf_dataset_split, vocab, tokenizer, transform=False):
        """
        Initialize the dataset.

        Args:
            hf_dataset_split (Dataset): Hugging Face dataset split with images and captions.
            vocab (object): Vocabulary for word-to-index conversion.
            tokenizer (object): Tokenizer for processing captions.
            transform (callable, optional): Transformations for images. Defaults to False.
        """
        self.dataset = hf_dataset_split
        self.transform = transform
        self.tokenizer = tokenizer
        self.vocab = vocab

    def __len__(self):
        """
        Return the total number of image-caption pairs.

        Returns:
            int: Dataset size.
        """
        return len(self.dataset) * 5

    def __getitem__(self, idx):
        """
        Retrieve an image-caption pair.

        Args:
            idx (int): Index of the item.

        Returns:
            tuple: Transformed image and tokenized caption.
        """
        image_idx = idx // 5
        caption_idx = idx % 5
        datapoint = self.dataset[image_idx]

        image = datapoint["image"]
        caption = datapoint["caption"][caption_idx]
        caption = self.tokenizer.tokenize(caption.lower())
        caption_final = [2] + [self.vocab.word_to_index(word) for word in caption] + [3]

        if self.transform:
            image = self.transform(image)

        return image, caption_final



