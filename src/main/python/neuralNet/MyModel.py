import torch
import torchvision.models as models
from torch.nn import Linear
from src.main.resources.CreateLogger import CreateLogger

# Initialize the logger
create_logger = CreateLogger("MyModel")
logger = create_logger.return_logger()


class MyModel:
    """
    Load a ResNet-50 model and replace the final layer for LBC loss.

    Args:
        n_categories (int): Number of categories/grid points. This implies
                            the number of grid separators is `n_categories - 1`.

    Attributes:
        model (torch.nn.Module): The modified ResNet-50 model.
        num_ftrs (int): The number of input features for the final layer.
        n_separators (int): The number of grid separators for the final layer.
    """

    def __init__(self, n_categories=130):
        """
        Initializes the MyModel class by loading a pretrained ResNet-50 model
        and replacing its final fully connected layer.

        Args:
            n_categories (int, optional): Number of categories/grid points. Default is 130.

        Raises:
            RuntimeError: If there is an error during model loading or layer replacement.
        """
        logger.info("Initializing MyModel starts")
        try:
            # Load the pretrained ResNet-50 model
            self.model = models.resnet50(pretrained=True)

            # Get the number of input features for the final layer
            self.num_ftrs = self.model.fc.in_features

            # Calculate the number of separators
            self.n_separators = n_categories - 1

            # Replace the final fully connected layer with a new layer
            self.model.fc = Linear(self.num_ftrs, self.n_separators)
        except Exception as e:
            logger.error(f"Error initializing MyModel: {e}")
            raise RuntimeError(f"Error initializing MyModel: {e}")
        logger.info("Initializing MyModel ends")

    def __call__(self):
        """
        Returns the modified ResNet-50 model.

        Returns:
            torch.nn.Module: The ResNet-50 model with the modified final layer.

        Raises:
            RuntimeError: If there is an issue returning the model.
        """
        logger.info("__call__ starts")
        try:
            return self.model
        except Exception as e:
            logger.error(f"Error in __call__: {e}")
            raise RuntimeError(f"Error in __call__: {e}")
