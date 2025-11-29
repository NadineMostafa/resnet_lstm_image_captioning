import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from ..data.dataset import ImageCaptioningDataset


def collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images)                       # preprocess returns tensor
    captions = [torch.tensor(c, dtype=torch.long) for c in captions]
    captions_padded = pad_sequence(captions, batch_first=True, padding_value=[0])
    return images, captions_padded

def generate_hf_splits(dataset):
    dataset = dataset.shuffle()
    dataset_portioned = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)

    dataset_validation_hf = dataset_portioned["test"]

    dataset_portioned = dataset_portioned["train"].train_test_split(test_size=0.125, shuffle=True, seed=42)

    dataset_train_hf = dataset_portioned["train"]
    dataset_test_hf = dataset_portioned["test"]

    return dataset_train_hf, dataset_validation_hf, dataset_test_hf


def generate_datasets(dataset_train_hf, dataset_validation_hf, dataset_test_hf, preprocess):
    train_dataset = ImageCaptioningDataset(dataset_train_hf, transform=preprocess)
    validation_dataset = ImageCaptioningDataset(dataset_validation_hf, transform=preprocess)
    test_dataset = ImageCaptioningDataset(dataset_test_hf, transform=preprocess)

    return train_dataset, validation_dataset, test_dataset

def create_data_loaders(train_dataset, validation_dataset,batch_size, collate_fn):
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=2, collate_fn=collate_fn, drop_last=True)
    validation_dataloader = DataLoader(validation_dataset, shuffle=True, batch_size=batch_size, num_workers=2, collate_fn=collate_fn, drop_last=True)

    return train_dataloader, validation_dataloader

def train_one_epoch(model, criterion, optimizer, train_dataloader, device, teacher_forcing_ratio, epoch, final_tf_ratio=0.7):
    if teacher_forcing_ratio > final_tf_ratio:  
        teacher_forcing_ratio = teacher_forcing_ratio - 0.025

    loss_epoch_train = 0.0
    num_batches_train = 0
    model.train()
      
    for images, captions in train_dataloader:
        images = images.to(device)
        captions = captions.to(device)
        optimizer.zero_grad()

        features = model.encoder(images)
        output = model.decoder(features, captions, teacher_forcing_ratio=teacher_forcing_ratio)
        
        targets = captions[:, 1:]
        
        loss = criterion(output.reshape(-1, output.size(2)), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        num_batches_train += 1
        loss_epoch_train += loss.item()
        print(f"epoch {epoch}, batch {num_batches_train}: loss = {loss.item()}")

    avg_loss_epoch_train = loss_epoch_train / num_batches_train
    print(f"Epoch {epoch} - Training Loss: {avg_loss_epoch_train:.4f}, TF Ratio: {teacher_forcing_ratio:.2f}")

    return avg_loss_epoch_train, teacher_forcing_ratio

def validate_one_epoch(model, criterion, validation_dataloader, device):
    model.decoder.eval()
    model.encoder.eval()

    loss_epoch_validation = 0
    num_batches_validation = 0
      
    with torch.no_grad():
      for images, captions in validation_dataloader:
        images = images.to(device)
        captions = captions.to(device)
        features = model.encoder(images)
        
        # Use teacher_forcing_ratio=1.0 for validation
        output = model.decoder(features, captions, teacher_forcing_ratio=1.0)
        
        targets = captions[:, 1:]
        loss = criterion(output.reshape(-1, output.size(2)), targets.reshape(-1))
        num_batches_validation += 1
        loss_epoch_validation += loss.item()
          
      loss_epoch_validation /= num_batches_validation
      print(f"Validation Loss: {loss_epoch_validation:.4f}")
  
    return loss_epoch_validation

def plot_losses(training_losses, validation_losses, path="./../outputs/training and validation losses.png"):
    epochs = range(1, len(training_losses) + 1)

    plt.plot(epochs, training_losses, 'b', label='Training loss')
    plt.plot(epochs, validation_losses, 'r', label='Validation loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(path)
    plt.close()

