from torch.utils.data import Dataset

class ImageCaptioningDataset(Dataset):
    def __init__(self, hf_dataset, vocab, tokenizer, transform=False):
        self.dataset = hf_dataset
        self.transform = transform
        self.tokenizer = tokenizer
        self.vocab = vocab

    def __len__(self):
        return len(self.dataset) * 5

    def __getitem__(self, idx):
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
        


