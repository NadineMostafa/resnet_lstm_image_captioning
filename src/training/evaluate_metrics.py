import torch
import evaluate
from datasets import load_dataset
from torchvision.models import ResNet50_Weights
from ..config import config
from ..utils.vocab import Vocab
from ..utils.utlis import generate_hf_splits
from ..models.Image_captioning_model import ImageCaptioningModel
from ..training.inference import generate_caption

def calculate_rouge(model, vocab, device, preprocess, test_split, output_path="./../outputs/rouge_results.txt"):
    """
    Calculate ROUGE scores for model-generated captions.

    Args:
        model (nn.Module): Trained image captioning model.
        vocab (Vocab): Vocabulary object for word-to-index mapping.
        device (torch.device): Device to run the model on (CPU/GPU).
        preprocess (callable): Preprocessing transformations for images.
        test_split (Dataset): Test dataset split.
        output_path (str, optional): Path to save the ROUGE results. Defaults to "./../outputs/rouge_results.txt".
    """
    model.eval()
    model.to(device)
    predictions = []
    refs = []
    for instance in test_split:
        caption = generate_caption(
            instance["image"],
            model,
            vocab.word2idx,
            vocab.idx2word,
            device,
            preprocess=preprocess
        )
        predictions.append(caption)
        refs.append(instance["caption"][0])

    print(f"finished inference")
    rouge = evaluate.load("rouge")

    results = rouge.compute(
    predictions=predictions,
    references=refs,
    use_stemmer=True
    )

    with open(output_path, "w") as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")

    print(results)


if __name__ == "__main__":
    """
    Main script to evaluate the model on the test dataset and calculate ROUGE scores.
    """
    dataset = load_dataset('nlphuji/flickr30k', split="test")
    _, _, test_split = generate_hf_splits(dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess = ResNet50_Weights.DEFAULT.transforms()

    print("Dataset loaded.")

    All_captions = dataset.data.column("caption").to_pylist()
    vocab = Vocab()
    vocab.build_vocab(All_captions)
    vocab.build_mappings(min_freq=config.VOCAB_MIN_FREQ)

    print("Vocabulary built.")
    print(f"Vocabulary size: {len(vocab)}")

    model = ImageCaptioningModel(embed_size=config.EMBED_SIZE, vocab_size=len(vocab), hidden_size=config.HIDDEN_SIZE)
    model.load_state_dict(torch.load("./../outputs/best_model.pt"))

    calculate_rouge(model,vocab, device, preprocess, test_split)
