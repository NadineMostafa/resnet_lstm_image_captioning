import os
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from ..config import config
from ..data.dataset import ImageCaptioningDataset


def collate_fn(batch):
    """
    Collate function to preprocess and pad a batch of image-caption pairs.

    Args:
        batch (list): A batch of tuples (image, caption).

    Returns:
        tuple: A tuple containing:
            - images (Tensor): Batch of images.
            - captions_padded (Tensor): Padded captions.
    """
    images, captions = zip(*batch)
    images = torch.stack(images)  # Stack images into a single tensor
    captions = [torch.tensor(c, dtype=torch.long) for c in captions]
    captions_padded = pad_sequence(captions, batch_first=True, padding_value=0)
    return images, captions_padded


def generate_hf_splits(dataset):
    """
    Generate train, validation, and test splits from a Hugging Face dataset.

    Args:
        dataset (Dataset): Hugging Face dataset.

    Returns:
        tuple: Train, validation, and test splits.
    """
    dataset = dataset.shuffle()
    dataset_portioned = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)

    dataset_validation_hf = dataset_portioned["test"]
    dataset_portioned = dataset_portioned["train"].train_test_split(test_size=0.125, shuffle=True, seed=42)

    dataset_train_hf = dataset_portioned["train"]
    dataset_test_hf = dataset_portioned["test"]

    return dataset_train_hf, dataset_validation_hf, dataset_test_hf


def generate_datasets(dataset_train_hf, dataset_validation_hf, dataset_test_hf, vocab, tokenizer, preprocess=False):
    """
    Generate PyTorch datasets for training, validation, and testing.

    Args:
        dataset_train_hf (Dataset): Training split.
        dataset_validation_hf (Dataset): Validation split.
        dataset_test_hf (Dataset): Test split.
        vocab (Vocab): Vocabulary object.
        tokenizer (Tokenizer): Tokenizer object.
        preprocess (callable, optional): Preprocessing transformations. Defaults to False.

    Returns:
        tuple: Training, validation, and test datasets.
    """
    train_dataset = ImageCaptioningDataset(dataset_train_hf, vocab, tokenizer, transform=preprocess)
    validation_dataset = ImageCaptioningDataset(dataset_validation_hf, vocab, tokenizer, transform=preprocess)
    test_dataset = ImageCaptioningDataset(dataset_test_hf, vocab, tokenizer, transform=preprocess)

    return train_dataset, validation_dataset, test_dataset


def create_data_loaders(train_dataset, validation_dataset, batch_size, collate_fn):
    """
    Create data loaders for training and validation datasets.

    Args:
        train_dataset (Dataset): Training dataset.
        validation_dataset (Dataset): Validation dataset.
        batch_size (int): Batch size.
        collate_fn (callable): Collate function for batching.

    Returns:
        tuple: Training and validation data loaders.
    """
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size, num_workers=2, collate_fn=collate_fn, drop_last=True
    )
    validation_dataloader = DataLoader(
        validation_dataset, shuffle=True, batch_size=batch_size, num_workers=2, collate_fn=collate_fn, drop_last=True
    )

    return train_dataloader, validation_dataloader


def train_one_epoch(model, criterion, optimizer, train_dataloader, device, teacher_forcing_ratio, epoch, final_tf_ratio=0.7):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The image captioning model.
        criterion (Loss): Loss function.
        optimizer (Optimizer): Optimizer for training.
        train_dataloader (DataLoader): DataLoader for the training dataset.
        device (torch.device): Device to run the model on (CPU/GPU).
        teacher_forcing_ratio (float): Initial teacher forcing ratio.
        epoch (int): Current epoch number.
        final_tf_ratio (float, optional): Final teacher forcing ratio. Defaults to 0.7.

    Returns:
        tuple: Average training loss and updated teacher forcing ratio.
    """
    # Adjust teacher forcing ratio
    initial_teacher_forcing = 1.0
    if teacher_forcing_ratio > final_tf_ratio:
        teacher_forcing_ratio = initial_teacher_forcing - (0.025 * epoch)

    loss_epoch_train = 0.0
    num_batches_train = 0
    model.train()

    print(f"Teacher forcing ratio: {teacher_forcing_ratio}")

    for images, captions in train_dataloader:
        images = images.to(device)
        captions = captions.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        features = model.encoder(images)
        output = model.decoder(features, captions, teacher_forcing_ratio=teacher_forcing_ratio)

        # Compute loss
        targets = captions[:, 1:]  # Shift captions for teacher forcing
        loss = criterion(output.reshape(-1, output.size(2)), targets.reshape(-1))

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Track loss
        num_batches_train += 1
        loss_epoch_train += loss.item()
        print(f"Epoch {epoch}, Batch {num_batches_train}: Loss = {loss.item()}")

    avg_loss_epoch_train = loss_epoch_train / num_batches_train
    print(f"Epoch {epoch} - Training Loss: {avg_loss_epoch_train:.4f}, TF Ratio: {teacher_forcing_ratio:.2f}")

    return avg_loss_epoch_train, teacher_forcing_ratio


def validate_one_epoch(model, criterion, validation_dataloader, device):
    """
    Validate the model for one epoch.

    Args:
        model (nn.Module): The image captioning model.
        criterion (Loss): Loss function.
        validation_dataloader (DataLoader): DataLoader for the validation dataset.
        device (torch.device): Device to run the model on (CPU/GPU).

    Returns:
        float: Average validation loss.
    """
    model.decoder.eval()
    model.encoder.eval()

    loss_epoch_validation = 0
    num_batches_validation = 0

    with torch.no_grad():
        for images, captions in validation_dataloader:
            images = images.to(device)
            captions = captions.to(device)

            # Forward pass
            features = model.encoder(images)
            output = model.decoder(features, captions, teacher_forcing_ratio=1.0)

            # Compute loss
            targets = captions[:, 1:]
            loss = criterion(output.reshape(-1, output.size(2)), targets.reshape(-1))

            # Track loss
            num_batches_validation += 1
            loss_epoch_validation += loss.item()

    avg_loss_epoch_validation = loss_epoch_validation / num_batches_validation
    print(f"Validation Loss: {avg_loss_epoch_validation:.4f}")

    return avg_loss_epoch_validation


def plot_losses(training_losses, validation_losses, parent_dir=config.OUTPUT_DIR):
    """
    Plot training and validation losses over epochs.

    Args:
        training_losses (list): List of training losses for each epoch.
        validation_losses (list): List of validation losses for each epoch.
        path (str, optional): Path to save the plot. Defaults to os.path.join(config.OUTPUT_DIR, "training_and_validation_losses.png").
    """
    epochs = range(1, len(training_losses) + 1)

    plt.plot(epochs, training_losses, 'b', label='Training loss')
    plt.plot(epochs, validation_losses, 'r', label='Validation loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(parent_dir, "training_and_validation_losses.png"))
    plt.close()

