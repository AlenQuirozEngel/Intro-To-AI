import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Custom Dataset Class for Loading MNIST from CSV Files
class MNISTDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.data = pd.read_csv(file_path)
        if self.data.shape[1] == 785:  # Check for training data
            self.X = self.data.iloc[:, 1:].values.astype(np.float32)  # Features (pixel data)
        else:  # For test data with no labels
            self.X = self.data.iloc[:, :].values.astype(np.float32)  # Only pixel data
        self.y = self.data.iloc[:, 0].values.astype(np.int64)  # Labels (digits)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = self.X[idx].reshape(28, 28)  # Reshape flat vector to 28x28 image
        label = self.y[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    

# Define the Neural Network Model
class MLP(nn.Module):
    def __init__(self, input_size=28*28, hidden_size=128, output_size=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the image
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Trainer Class to Handle Training and Validation
class NeuralNetworkTrainer:
    def __init__(self, train_file, test_file, batch_size=32, hidden_size=64, output_size=10, lr=0.0001, epochs=20):
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_accuracy_list = []  # To track training accuracy per epoch
        self.val_accuracy_list = []  # To track validation accuracy per epoch

        # Data transforms (normalize pixel values between 0 and 1)
        transform = transforms.Compose([transforms.ToTensor()])

        # Load training and test datasets
        self.train_dataset = MNISTDataset(train_file, transform=transform)
        self.test_dataset = MNISTDataset(test_file, transform=transform)

        # Create data loaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

        # Initialize the MLP model
        self.model = MLP(hidden_size=hidden_size, output_size=output_size)
        
        # Loss function (CrossEntropyLoss is commonly used for classification tasks)
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer (Stochastic Gradient Descent)
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        self.fc1 = nn.Sequential(
            nn.Linear(28 * 28, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout added to prevent overfitting
            nn.BatchNorm1d(hidden_size)  # Batch normalization added
        )



    def train(self):
        self.model.train()  # Set the model to training mode
        for epoch in range(self.epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in self.train_loader:
                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                running_loss += loss.item()

            accuracy = 100 * correct / total
            self.train_accuracy_list.append(accuracy)  # Track training accuracy
            print(f'Epoch [{epoch + 1}/{self.epochs}], Loss: {running_loss/len(self.train_loader):.4f}, Training Accuracy: {accuracy:.2f}%')

            # Validate after each epoch and track validation accuracy
            val_accuracy = self.validate()
            self.val_accuracy_list.append(val_accuracy)

    def validate(self):
        self.model.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():  # Disable gradient computation during validation
            for images, labels in self.test_loader:
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Validation Accuracy: {accuracy:.2f}%')
        return accuracy

    def plot_confusion_matrix(self):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in self.test_loader:
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Compute confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

    def plot_accuracy(self):
        # Plot Training Accuracy and Validation Accuracy vs Epochs
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, self.epochs + 1), self.train_accuracy_list, marker='o', linestyle='-', color='b', label='Training Accuracy')
        plt.plot(range(1, self.epochs + 1), self.val_accuracy_list, marker='x', linestyle='--', color='r', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def run(self):
        # Train the model
        self.train()

        # Plot accuracy over epochs
        self.plot_accuracy()

        # Plot confusion matrix
        self.plot_confusion_matrix()

# Entry point for training and validating the neural network
if __name__ == "__main__":
    trainer = NeuralNetworkTrainer(train_file='C:/Users/alenq/Documents/Computer_Science_Course_UM/repository year 2/AI Wee 3/__pycache__/train.csv', test_file='C:/Users/alenq/Documents/Computer_Science_Course_UM/repository year 2/AI Wee 3/__pycache__/test.csv', batch_size=64, hidden_size=128, output_size=10, lr=0.01, epochs=20)
    trainer.run()
