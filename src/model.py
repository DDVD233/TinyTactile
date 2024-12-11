import torch
import torch.nn as nn
from torchvision.models import resnet18
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np


class ShallowCNN(nn.Module):
    def __init__(self, num_classes):
        super(ShallowCNN, self).__init__()
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32),

            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
        )

        self.flatten_size = 64 * 8 * 8

        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class ModelFactory:
    @staticmethod
    def create_model(model_name, num_classes, device):
        if model_name == 'resnet':
            model = resnet18(pretrained=True)
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model.to(device)

        elif model_name == 'shallow_cnn':
            return ShallowCNN(num_classes).to(device)

        elif model_name == 'knn':
            return KNeighborsClassifier(n_neighbors=3)

        elif model_name == 'svm':
            return SVC(kernel='rbf')

        else:
            raise ValueError(f"Unknown model type: {model_name}")


def train_model(model, train_loader, test_loader, device, model_type='deep', num_epochs=100):
    if model_type == 'deep':
        return train_deep_model(model, train_loader, test_loader, device, num_epochs)
    else:
        return train_sklearn_model(model, train_loader, test_loader)


def train_deep_model(model, train_loader, test_loader, device, num_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    best_acc = 0.0

    for epoch in range(num_epochs):
        # Training
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

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total

        # Evaluation
        test_acc = evaluate_deep_model(model, test_loader, device)

        print(f'Epoch [{epoch + 1}/{num_epochs}]:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Test Acc: {test_acc:.2f}%')

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f'models/best_{model.__class__.__name__}_model.pth')

    return best_acc


def train_sklearn_model(model, train_loader, test_loader):
    # Convert PyTorch DataLoader to numpy arrays
    X_train, y_train = [], []
    X_test, y_test = [], []

    for inputs, labels in train_loader:
        X_train.append(inputs.numpy().reshape(inputs.shape[0], -1))
        y_train.append(labels.numpy())

    for inputs, labels in test_loader:
        X_test.append(inputs.numpy().reshape(inputs.shape[0], -1))
        y_test.append(labels.numpy())

    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)
    X_test = np.vstack(X_test)
    y_test = np.concatenate(y_test)

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate
    train_acc = 100 * model.score(X_train, y_train)
    test_acc = 100 * model.score(X_test, y_test)

    print(f'Train Accuracy: {train_acc:.2f}%')
    print(f'Test Accuracy: {test_acc:.2f}%')

    return test_acc


def evaluate_deep_model(model, test_loader, device):
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