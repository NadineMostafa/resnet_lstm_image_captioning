import torch
import torch.nn as nn
from datasets import load_dataset
from ..config import config
from ..utils.vocab import Vocab
from ..utils.utlis import generate_hf_splits, generate_datasets, collate_fn, create_data_loaders, train_one_epoch, validate_one_epoch, plot_losses
from ..models.Image_captioning_model import ImageCaptioningModel
from torchvision.models import ResNet50_Weights
from .evaluate_metrics import calculate_rouge

dataset = load_dataset('nlphuji/flickr30k', split="test")
print("Dataset loaded.")

All_captions = dataset.data.column("caption").to_pylist()
vocab = Vocab(All_captions)
vocab.build_vocab(All_captions)
vocab.build_mappings(min_freq=config.VOCAB_MIN_FREQ)

print("Vocabulary built.")
print(f"Vocabulary size: {len(vocab)}")


model = ImageCaptioningModel(embed_size=config.EMBED_SIZE, vocab_size=len(vocab), hidden_size=config.HIDDEN_SIZE)
print("Model initialized.")


for param in model.encoder.CNNBackbone.parameters():
    param.requires_grad = False

for param in model.encoder.CNNBackbone[-1].parameters():
    param.requires_grad = True

preprocess = ResNet50_Weights.DEFAULT.transforms()

dataset_train_hf, dataset_validation_hf, dataset_test_hf = generate_hf_splits(dataset)
train_dataset, validation_dataset, test_dataset = generate_datasets(dataset_train_hf, dataset_validation_hf, dataset_test_hf, preprocess=preprocess)
train_dataloader, validation_dataloader = create_data_loaders(train_dataset, validation_dataset, config.BATCH_SIZE, collate_fn)

print("Data loaders created.")
print(f"Number of training batches: {len(train_dataloader)}")


optimizer = torch.optim.Adam(
    list(filter(lambda p: p.requires_grad, model.encoder.parameters())) + list(model.decoder.parameters()),
    lr=config.LEARNING_RATE
)

criterion = nn.CrossEntropyLoss(ignore_index=0)
print("Optimizer and loss function set up.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


training_loss_epochs = list()
validation_loss_epochs = list()

model = model.to(device)
teacher_forcing_ratio = 1
best_val_loss = float('inf')
epochs_no_improve = 0
best_model_path = "./../outputs/best_model.pt"


for epoch in range(1, config.NUM_EPOCHS + 1):
    train_loss, teacher_forcing_ratio = train_one_epoch(model, criterion, optimizer, train_dataloader, device, teacher_forcing_ratio, epoch)
    val_loss = validate_one_epoch(model, criterion, validation_dataloader, device)
    training_loss_epochs.append(train_loss)
    validation_loss_epochs.append(val_loss)

    
    # Early stopping logic
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      epochs_no_improve = 0
      torch.save(model.state_dict(), best_model_path)
      print(f"✓ Validation loss improved! Model saved to {best_model_path}")
    else:
      epochs_no_improve += 1
      print(f"✗ No improvement for {epochs_no_improve} epoch(s)")
      
      if epochs_no_improve >= 3:
        print(f"\nEarly stopping triggered after {epoch + 1} epochs!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        break
      


print("Training complete.")
print(f"Training losses over epochs: {training_loss_epochs}")
print(f"Validation losses over epochs: {validation_loss_epochs}")
plot_losses(training_loss_epochs, validation_loss_epochs, path="./../outputs/loss_plot.png")
print(f"Best model saved at: {best_model_path}")

calculate_rouge(model, dataset_test_hf, output_path="./../outputs/rouge_results.txt")


