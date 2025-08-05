import torch
from main.resources.CreateLogger import CreateLogger

# Create a logger instance
create_logger = CreateLogger("LBCLabel")
logger = create_logger.return_logger()

class LBCLabel:
    """
    A class to convert label numbers into label vectors for multi-task learning by confusion (LBC).

    This class transforms sample labels represented as numbers into label vectors suitable for
    training tasks that involve subsets of categories.
    """

    def __init__(self, subset, device='cuda'):
        """
        Initializes the LBCLabel class.

        Args:
            subset (list or torch.Tensor):
                A list or tensor of indices representing the subset of categories.
            device (str, optional):
                The device to perform computations on ('cuda' or 'cpu'). Default is 'cuda'.

        Raises:
            ValueError: If the subset is empty or invalid.
        """
        logger.info("Initializing LBCLabel starts")

        # Validate and set the subset
        if subset is None or len(subset) == 0:
            logger.error("Subset is None or empty.")
            raise ValueError("Subset cannot be None or empty.")
        self.subset = subset

        # Validate and set the device
        try:
            self.device = torch.device(device)
        except Exception as e:
            logger.error(f"Invalid device specified: {device}. Error: {e}")
            raise ValueError(f"Invalid device: {device}") from e

        logger.info("Initialized LBCLabel ends")

    def __call__(self, y, return_as_float=False):
        """
        Converts label numbers into label vectors for the specified subset of categories.

        Args:
            y (torch.Tensor): Labels of the samples as numbers.
            return_as_float (bool, optional): If True, returns the result as a float tensor. Default is False.

        Returns:
            torch.Tensor: A tensor indicating which category indices in `subset` are less than each label in `y`.

        Raises:
            ValueError: If `y` is invalid.
            RuntimeError: If tensor operations fail.
        """
        logger.info("__call__ starts")

        # Validate input `y`
        if y is None:
            logger.error("Input y is None.")
            raise ValueError("Input y cannot be None.")
        if not isinstance(y, torch.Tensor):
            logger.error("Input y is not a torch.Tensor.")
            raise ValueError("Input y must be a torch.Tensor.")
        if y.dim() != 1:
            logger.error(f"Input y has invalid dimensions: {y.dim()}")
            raise ValueError("Input y must be a 1D torch.Tensor.")

        try:
            # Convert subset to a tensor on the specified device
            subset_tensor = torch.tensor(self.subset, device=self.device).view(1, -1)

            # Ensure `y` is on the same device
            y = y.to(self.device)

            # Compare the subset tensor with the label tensor
            result = subset_tensor < y.view(-1, 1)

            # Optionally return as float for compatibility
            if return_as_float:
                result = result.float()

            logger.debug(f"Converted labels to vectors: {result}")
            logger.info("__call__ ends")
            return result

        except Exception as e:
            logger.error(f"Error during label vector conversion: {e}")
            raise RuntimeError("Failed to convert label numbers into label vectors.") from e
