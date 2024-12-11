import torch
import torch.nn as nn
import torch.nn.functional as F
from src.main.resources.CreateLogger import CreateLogger

create_logger = CreateLogger("SimpleCNN")
logger = create_logger.return_logger()


class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network (CNN) for image classification.

    This network consists of two convolutional layers followed by two fully connected layers.
    It uses average pooling for downsampling and ReLU activations for non-linearity.

    Attributes:
        img_dim (int): The height and width of the input image (assumed square).
        n_hidden (int): The number of hidden units in the first fully connected layer.
        n_kernels (int): The number of convolutional kernels (filters) in the first convolutional layer.
        pool (nn.AvgPool2d): The average pooling layer used for downsampling.
        conv1 (nn.Conv2d): The first convolutional layer.
        conv2 (nn.Conv2d): The second convolutional layer.
        fc1 (nn.Linear): The first fully connected layer.
        fc2 (nn.Linear): The second fully connected layer that outputs class scores.
    """

    def __init__(self, img_dim=60, n_categories=200, n_hidden=128, n_kernels=16):
        """
        Initializes the SimpleCNN model.

        Args:
            img_dim (int, optional): The height and width of the input image (default is 60).
            n_categories (int, optional): The number of output categories/classes (default is 200).
            n_hidden (int, optional): The number of hidden units in the first fully connected layer (default is 128).
            n_kernels (int, optional): The number of kernels (filters) in the first convolutional layer (default is 16).
        """
        logger.info("SimpleCNN model start initialisation")
        super(SimpleCNN, self).__init__()
        try:
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
            self.img_dim = img_dim
            self.n_hidden = n_hidden
            self.n_kernels = n_kernels

            self.conv1 = nn.Conv2d(1, self.n_kernels, kernel_size=2, stride=1, padding=1)
            self.conv2 = nn.Conv2d(self.n_kernels, 2 * self.n_kernels, kernel_size=2, stride=1, padding=1)
            self.fc1 = nn.Linear(2 * self.n_kernels * (self.img_dim // 4) * (self.img_dim // 4), self.n_hidden)
            self.fc2 = nn.Linear(self.n_hidden, n_categories - 1)
        except Exception as e:
            logger.error(f"Error initializing the model: {e}")
            raise RuntimeError("Model initialization failed.") from e
        logger.info("SimpleCNN model initialized")

    def forward(self, x):
        """
        Defines the forward pass of the SimpleCNN model.

        Args:
            x (torch.Tensor): A batch of input images with shape (batch_size, 1, img_dim, img_dim).

        Returns:
            torch.Tensor: The output logits for each class, with shape (batch_size, n_categories - 1).
        """
        logger.info("Model forward starts")
        try:
            x = self.pool(F.relu(self.conv1(x)))  # Apply first convolution, ReLU, and pooling
            x = self.pool(F.relu(self.conv2(x)))  # Apply second convolution, ReLU, and pooling
            x = x.view(-1, 2 * self.n_kernels * (self.img_dim // 4) * (self.img_dim // 4))  # Flatten the tensor
            x = F.relu(self.fc1(x))  # Apply first fully connected layer with ReLU
            x = self.fc2(x)          # Apply second fully connected layer (output layer)
            logger.info("Model forward pass is done")
            return x
        except Exception as e:
            logger.error(f"Error during the forward pass: {e}")
            raise RuntimeError("Forward pass failed.") from e

