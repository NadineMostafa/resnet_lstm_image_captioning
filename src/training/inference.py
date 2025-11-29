import torch
from PIL import Image
from datasets import load_dataset
from ..config import config
from ..utils.vocab import Vocab
from torchvision.models import ResNet50_Weights
from ..models.Image_captioning_model import ImageCaptioningModel

def generate_caption(image, model, word2idx, idx2word, device, max_length=50, preprocess=None):
    model.eval()
    
    if preprocess:
        image = preprocess(image)
    
    if image.dim() == 3:
        image = image.unsqueeze(0)
    image = image.to(device)
    
    with torch.no_grad():
        features = model.encoder(image)  # (1, embed_size)
        hidden_state = None
        input_token = features.unsqueeze(1)  # (1, 1, embed_size)        
        seq = [word2idx['<SOS>']]
        
        for _ in range(max_length):
            lstm_out, hidden_state = model.decoder.lstm(input_token, hidden_state)
            logits = model.decoder.linear(lstm_out)  # (1, 1, vocab_size)
            next_idx = int(torch.argmax(logits[0, -1, :]).item())
            
            if next_idx == word2idx['<EOS>']:
                break
            
            seq.append(next_idx)
            
            input_token = model.decoder.embedding(torch.tensor([[next_idx]], device=device))
        
        words = []
        for idx in seq[1:]:  # Skip <SOS>
            if idx == word2idx['<EOS>']:
                break
            words.append(idx2word.get(idx, "<UNK>"))
        
        return " ".join(words)
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = load_dataset('nlphuji/flickr30k', split="test")
    print("Dataset loaded.")

    All_captions = dataset.data.column("caption").to_pylist()
    vocab = Vocab()
    vocab.build_vocab(All_captions)
    vocab.build_mappings(min_freq=config.VOCAB_MIN_FREQ)

    print("Vocabulary built.")
    print(f"Vocabulary size: {len(vocab)}")

    model = ImageCaptioningModel(embed_size=config.EMBED_SIZE, vocab_size=len(vocab), hidden_size=config.HIDDEN_SIZE)
    model.load_state_dict(torch.load("./../outputs/best_model.pt"))
    model = model.to(device)

    preprocess = ResNet50_Weights.DEFAULT.transforms()

    image_path = ""
    if image_path:
        image = Image.open(image_path).convert("RGB")
    else: image = dataset[0]["image"]

    genereated_caption = generate_caption(image, model, vocab.word2idx, vocab.idx2word, device, preprocess=preprocess)
    print(f"Generated Caption: {genereated_caption}")

