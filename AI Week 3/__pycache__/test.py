import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Flatten the input from (batch_size, 28, 28) to (batch_size, 784)
        x = x.view(x.size(0), -1)  # Flatten the image
        x = F.relu(self.hidden_layer(x))  # Apply ReLU to the hidden layer
        x = self.output_layer(x)  # Linear layer output (no activation for logits)
        return x

class NeuralNetworkTrainer:
    def __init__(self, train_file, test_file, batch_size=64, hidden_dim=128, output_dim=10, lr=0.01, epochs=20):
        self.batch_size = batch_size
        self.epochs = epochs

        # Load train and test datasets using Pandas
        train_data = pd.read_csv(train_file)
        test_data = pd.read_csv(test_file)

        # Extract image data and labels for training data
        mnist_train_x = train_data.iloc[:, 1:].values.astype('float32') / 255.0  # Normalize the data to [0, 1]
        mnist_train_y = train_data.iloc[:, 0].values.astype('int64')

        # Extract image data and labels for test data (validation set)
        mnist_test_x = test_data.iloc[:, 1:].values.astype('float32') / 255.0
        mnist_test_y = test_data.iloc[:, 0].values.astype('int64')

        # Convert data to PyTorch tensors
        self.train_dataset = TensorDataset(torch.tensor(mnist_train_x), torch.tensor(mnist_train_y))
        self.val_dataset = TensorDataset(torch.tensor(mnist_test_x), torch.tensor(mnist_test_y))

        # Initialize the network
        self.network = MLP(28 * 28, hidden_dim, output_dim)

        # Initialize the optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        # Initialize the loss function
        self.loss_func = nn.CrossEntropyLoss()

    def train(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

        val_accuracy_list = np.zeros(self.epochs)

        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")

            # Training loop
            self.network.train()  # Set the network in training mode
            for batch_idx, (digits, labels) in enumerate(train_loader):
                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.network(digits)

                # Compute loss
                loss = self.loss_func(outputs, labels)

                # Backward pass
                loss.backward()

                # Update weights
                self.optimizer.step()

            # Validate the model
            accuracy = self.validate(val_loader)
            val_accuracy_list[epoch] = accuracy
            print(f"Validation Accuracy: {accuracy * 100:.2f}%")

        return val_accuracy_list

    def validate(self, val_loader):
        self.network.eval()  # Set the network in evaluation mode
        correct = 0
        total = 0

        with torch.no_grad():  # Disable gradient computation
            for digits, labels in val_loader:
                outputs = self.network(digits)
                predicted = torch.argmax(outputs, dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        return correct / total

    def confusion_matrix(self):
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for digits, labels in val_loader:
                outputs = self.network(digits)
                predicted = torch.argmax(outputs, dim=1)
                all_preds.extend(predicted.numpy())
                all_labels.extend(labels.numpy())

        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

    def run(self):
        # Train the model and capture validation accuracy per epoch
        val_accuracy_list = self.train()

        # Plot the validation accuracy
        plt.figure()
        plt.plot([i + 1 for i in range(self.epochs)], val_accuracy_list, 'o-')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Accuracy')
        plt.title('Validation Accuracy per Epoch')
        plt.grid(True)
        plt.show()

        # Display the confusion matrix for the validation set
        print("Confusion Matrix on Validation Set:")
        self.confusion_matrix()


if __name__ == "__main__":
    trainer = NeuralNetworkTrainer(train_file='train.csv', test_file='test.csv', batch_size=64, hidden_dim=128, output_dim=10, lr=0.001, epochs=20)
    trainer.run()
