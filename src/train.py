import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from sklearn.model_selection import train_test_split
from utils import read_key_data


class TactileDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


def preprocess_data(data_dict):
    samples = []
    labels = []
    label_to_idx = {key: idx for idx, key in enumerate(sorted(data_dict.keys()))}

    for key, tactile_readings in data_dict.items():
        for reading in tactile_readings:
            # Take min over temporal dimension
            processed = np.min(reading, axis=0)
            # Normalize to [0, 1]
            processed = (processed - processed.min()) / (processed.max() - processed.min())
            samples.append(processed[np.newaxis, :, :])  # Add channel dimension
            labels.append(label_to_idx[key])

    return np.array(samples), np.array(labels), label_to_idx


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device).float(), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(train_loader), 100. * correct / total


def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device).float(), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return 100. * correct / total


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def combine_data_dicts(file_list, data_folder):
    """
    Combine multiple HDF5 files into a single data dictionary.

    Args:
        file_list (list): List of HDF5 filenames
        data_folder (str): Path to the data folder

    Returns:
        combined_dict (dict): Combined dictionary with all tactile readings
    """
    # Initialize combined dictionary using defaultdict to automatically handle new keys
    combined_dict = defaultdict(list)

    # Process each file
    for file in file_list:
        path = os.path.join(data_folder, file)
        data = read_key_data(path)

        # Extend the combined dictionary with new readings
        for key, readings in data.items():
            combined_dict[key].extend(readings)

    # Convert defaultdict back to regular dict
    return dict(combined_dict)


def preprocess_combined_data(data_dict):
    """
    Preprocess the combined data dictionary.

    Args:
        data_dict (dict): Combined dictionary with all tactile readings

    Returns:
        tuple: (processed samples, labels, label mapping)
    """
    samples = []
    labels = []
    label_to_idx = {key: idx for idx, key in enumerate(sorted(data_dict.keys()))}

    for key, tactile_readings in data_dict.items():
        for reading in tactile_readings:
            # Take min over temporal dimension
            processed = np.min(reading, axis=0)
            # Normalize to [0, 1]
            processed = (processed - processed.min()) / (processed.max() - processed.min())
            samples.append(processed[np.newaxis, :, :])  # Add channel dimension
            labels.append(label_to_idx[key])

    return np.array(samples), np.array(labels), label_to_idx


def train():
    # Device configuration
    device = get_device()

    # Load and preprocess data
    data_folder = 'recordings'
    file_list = [f for f in os.listdir(data_folder)
                 if f.endswith('layout_1.hdf5') or f.endswith('2024-11-25_18-13-21.hdf5')]

    combined_data = combine_data_dicts(file_list, data_folder)
    samples, labels, label_to_idx = preprocess_combined_data(combined_data)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        samples, labels, test_size=0.3, random_state=42, stratify=labels
    )

    # Create datasets and dataloaders
    train_dataset = TactileDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_dataset = TactileDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model
    model = resnet18(pretrained=True)
    # Modify first conv layer to accept 1 channel instead of 3
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # Modify final layer to match number of classes
    num_classes = len(label_to_idx)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

    # Training loop
    num_epochs = 100
    best_acc = 0.0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_acc = evaluate(model, test_loader, device)

        print(f'Epoch [{epoch + 1}/{num_epochs}]:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Test Acc: {test_acc:.2f}%')

        if test_acc > best_acc:
            best_acc = test_acc
            # Save best model
            torch.save(model.state_dict(), 'best_tactile_model.pth')

    print(f'\nBest Test Accuracy: {best_acc:.2f}%')
    print('\nClass mapping:')
    for key, idx in label_to_idx.items():
        print(f'{key}: {idx}')


if __name__ == '__main__':
    train()