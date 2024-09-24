import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from models import CoNvNet
from create_dataset import CreateDataset
import torch.optim as optim
import pandas as pd

class CancerClassifier:
    def __init__(self, num_classes=44, lr=0.001, batch_size=32, num_epochs=10):
        """
        Initialize the CancerClassifier.

        Parameters:
        - num_classes (int): Number of output classes.
        - lr (float): Learning rate for the optimizer.
        - batch_size (int): Batch size for data loaders.
        - num_epochs (int): Number of training epochs.
        """
        self.num_classes = num_classes
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = CoNvNet(num_classes=num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.train_loader, self.test_loader, self.valid_loader = self._prepare_data()

    def _prepare_data(self):
        """
        Prepare data loaders for training, testing, and validation.
        """
        df = pd.read_csv("dataset.csv")
        df = df[["file_path", "class"]]

        transform = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor()])

        train_dataset = CreateDataset(dataframe=df, transform=transform, split_type='train')
        test_dataset = CreateDataset(dataframe=df, transform=transform, split_type='test')
        valid_dataset = CreateDataset(dataframe=df, transform=transform, split_type='valid')

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader, valid_loader

    def train(self):
        """
        Train the CancerClassifier.
        """
        epoch_train_loss = []
        epoch_validation_loss = []
        epoch_train_accuracy = []
        epoch_validation_accuracy = []

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0.0
            correct_train = 0
            total_train = 0

            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                _, predicted_train = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted_train == labels).sum().item()

                progress_bar.set_postfix(train_loss=loss.item(), train_acc=correct_train / total_train)

            average_loss = total_loss / len(self.train_loader)
            train_accuracy = correct_train / total_train

            epoch_train_accuracy.append(train_accuracy)
            epoch_train_loss.append(average_loss)

            # Validation
            self.model.eval()
            correct_val = 0
            total_val = 0
            total_val_loss = 0.0

            with torch.no_grad():
                for inputs, labels in self.valid_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs_val = self.model(inputs)
                    loss_val = self.criterion(outputs_val, labels)
                    total_val_loss += loss_val.item()

                    _, predicted_val = torch.max(outputs_val.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted_val == labels).sum().item()

                average_val_loss = total_val_loss / len(self.valid_loader)
                val_accuracy = correct_val / total_val

            epoch_validation_accuracy.append(val_accuracy)
            epoch_validation_loss.append(average_val_loss)

            tqdm.write(f"Validation_acc={val_accuracy:.4f}, validation_loss={average_val_loss:.4f}")

        pd.DataFrame(
            list(zip(epoch_train_loss, epoch_validation_loss, epoch_train_accuracy, epoch_validation_accuracy)),
            columns=["TL", "VL", "TA", "VA"]).to_csv("performance.csv", index=False)

        # testing
        self.model.eval()
        correct_test = 0
        total_test = 0
        actual_class = []
        predicted_class = []

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs_test = self.model(inputs)
                _, predicted_test = torch.max(outputs_test.data, 1)

                actual_class += list(labels.cpu().numpy())
                predicted_class += list(predicted_test.cpu().numpy())

                total_test += labels.size(0)
                correct_test += (predicted_test == labels).sum().item()

        accuracy_test = correct_test / total_test
        tqdm.write(f"Test Accuracy: {accuracy_test:.4f}")

        pd.DataFrame(list(zip(actual_class, predicted_class)), columns=["AC", "PC"]).to_csv("results.csv", index=False)

        torch.save(self.model.state_dict(), "cancer_classifier_model.pth")

if __name__ == "__main__":
    classifier = CancerClassifier()
    classifier.train()
