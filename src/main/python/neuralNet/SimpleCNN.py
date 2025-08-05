import torch
import torch.nn as nn
import torch.nn.functional as F
from main.resources.CreateLogger import CreateLogger

create_logger = CreateLogger("SimpleCNN")
logger = create_logger.return_logger()


class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network (CNN) for image classification.

    The network consists of two convolutional layers followed by two fully connected layers.
    Average pooling is used for downsampling, and ReLU activations introduce non-linearity.

    Args:
        img_dim (int): The height and width of the input image (assumed square).
        n_categories (int): The number of output categories/classes.
        n_hidden (int): The number of hidden units in the first fully connected layer.
        n_kernels (int): The number of convolutional kernels (filters) in the first convolutional layer.

    Attributes:
        conv1 (nn.Conv2d): The first convolutional layer.
        conv2 (nn.Conv2d): The second convolutional layer.
        pool (nn.AvgPool2d): The average pooling layer for downsampling.
        fc1 (nn.Linear): The first fully connected layer.
        fc2 (nn.Linear): The second fully connected layer (output layer).
    """

    def __init__(self, img_dim=60, n_categories=200, n_hidden=128, n_kernels=16):
        """
        Initializes the SimpleCNN model.

        Args:
            img_dim (int): The height and width of the input image (assumed square).
            n_categories (int): The number of output categories/classes.
            n_hidden (int): The number of hidden units in the first fully connected layer.
            n_kernels (int): The number of convolutional kernels in the first convolutional layer.

        Raises:
            RuntimeError: If the model initialization fails.
        """
        logger.info("Initializing SimpleCNN model starts")
        super(SimpleCNN, self).__init__()

        try:
            self.img_dim = img_dim
            self.n_hidden = n_hidden
            self.n_kernels = n_kernels

            # Define layers
            self.pool = nn.AvgPool2d(2, 2)
            self.conv1 = nn.Conv2d(1, self.n_kernels, kernel_size=2, stride=1, padding=1)
            self.conv2 = nn.Conv2d(self.n_kernels, 2 * self.n_kernels, kernel_size=2, stride=1, padding=1)
            flattened_dim = 2 * self.n_kernels * (self.img_dim // 4) * (self.img_dim // 4)
            self.fc1 = nn.Linear(flattened_dim, self.n_hidden)
            self.fc2 = nn.Linear(self.n_hidden, n_categories - 1)

            logger.info(f"SimpleCNN initialized with img_dim={self.img_dim}, "
                        f"n_hidden={self.n_hidden}, n_kernels={self.n_kernels}")

        except Exception as e:
            logger.error(f"Error initializing SimpleCNN: {e}")
            raise RuntimeError("Model initialization failed.") from e

    def forward(self, x):
        """
        Defines the forward pass of the SimpleCNN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, img_dim, img_dim).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, n_categories - 1).

        Raises:
            RuntimeError: If the forward pass fails.
        """
        logger.info("Starting forward pass of SimpleCNN")

        # Validate input tensor
        if x.dim() != 4 or x.size(1) != 1 or x.size(2) != self.img_dim or x.size(3) != self.img_dim:
            logger.error(f"Invalid input tensor shape: {x.shape}. Expected (batch_size, 1, {self.img_dim}, {self.img_dim})")
            raise ValueError(f"Input tensor must have shape (batch_size, 1, {self.img_dim}, {self.img_dim})")

        try:
            # Convolutional layers
            x = self.pool(F.relu(self.conv1(x)))
            logger.debug(f"Shape after conv1: {x.shape}")
            x = self.pool(F.relu(self.conv2(x)))
            logger.debug(f"Shape after conv2: {x.shape}")

            # Flatten
            x = x.view(-1, 2 * self.n_kernels * (self.img_dim // 4) * (self.img_dim // 4))
            logger.debug(f"Shape after flattening: {x.shape}")

            # Fully connected layers
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            logger.info("Forward pass of SimpleCNN completed")
            return x
        except Exception as e:
            logger.error(f"Error during forward pass: {e}")
            raise RuntimeError("Forward pass failed.") from e
