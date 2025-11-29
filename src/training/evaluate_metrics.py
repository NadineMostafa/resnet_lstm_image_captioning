import torch
import evaluate
from datasets import load_dataset
from ..config import config
from ..utils.vocab import Vocab
from ..utils.utlis import generate_hf_splits
from ..models.Image_captioning_model import ImageCaptioningModel
from ..training.inference import generate_caption

def calculate_rouge(model, test_split, output_path="./../outputs/rouge_results.txt"):
    model.eval()
    predictions = list()
    refs = list()
    for instance in test_split:
        caption = generate_caption(instance["image"], model, trans=True)
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
    dataset = load_dataset('nlphuji/flickr30k', split="test")
    _, _, test_split = generate_hf_splits(dataset)

    print("Dataset loaded.")

    All_captions = dataset.data.column("caption").to_pylist()
    vocab = Vocab()
    vocab.build_vocab(All_captions)
    vocab.build_mappings(min_freq=config.VOCAB_MIN_FREQ)

    print("Vocabulary built.")
    print(f"Vocabulary size: {len(vocab)}")

    model = ImageCaptioningModel(embed_size=config.EMBED_SIZE, vocab_size=len(vocab), hidden_size=config.HIDDEN_SIZE)
    model.load_state_dict(torch.load("./../outputs/best_model.pt"))

    calculate_rouge(model, test_split)

