import torch
import torch.nn as nn

class CoNvNet(nn.Module):
    """
    Convolutional Neural Network (CoNvNet) model for image classification.

    Parameters:
    - num_classes (int, optional): Number of output classes. Default is 44.

    Attributes:
    - conv1, relu1, pool1: First convolutional layer with ReLU activation and max-pooling.
    - conv2, relu2, pool2: Second convolutional layer with ReLU activation and max-pooling.
    - conv3, relu3, pool3: Third convolutional layer with ReLU activation and max-pooling.
    - flatten: Flatten layer to convert the output from convolutional layers to a 1D tensor.
    - fc1, relu4, fc2: Fully connected layers for classification.
    """

    def __init__(self, num_classes=44):
        """
        Initialize the CoNvNet model with specified parameters.
        """
        super(CoNvNet, self).__init__()

        # Convolutional layers with ReLU activation and max-pooling
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layers for classification
        self.fc1 = nn.Linear(64 * 28 * 28, 512)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        """
        Forward pass of the CoNvNet model.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor.
        """
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)

        return x
