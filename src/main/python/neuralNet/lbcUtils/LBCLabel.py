import torch
from src.main.resources.CreateLogger import CreateLogger

# Create a logger instance
create_logger = CreateLogger("LBCLabel")
logger = create_logger.return_logger()

class LBCLabel:
    """
    A class to convert label numbers into label vectors for multi-task learning by confusion (LBC).

    This class transforms sample labels represented as numbers into label vectors suitable for
    training tasks that involve subsets of categories.

    Attributes:
        y (torch.Tensor): Labels of the samples as numbers (e.g., 0 for the first category).
        subset (list or torch.Tensor): A list or tensor of category indices for which to calculate the loss.
        device (str): The computation device ('cuda' or 'cpu').
    """

    def __init__(self, y, subset, device='cuda'):
        """
        Initializes the LBCLabel class.

        Args:
            y (torch.Tensor):
                Labels of the samples as numbers, e.g., 0 for the first category.
            subset (list or torch.Tensor):
                A list or tensor of indices representing the subset of categories.
            device (str, optional):
                The device to perform computations on ('cuda' or 'cpu'). Default is 'cuda'.
        """
        logger.info("Initializing LBCLabel starts")
        self.y = y
        self.subset = subset
        self.device = device
        logger.info("Initialized LBCLabel ends")

    def __call__(self):
        """
        Converts label numbers into label vectors for the specified subset of categories.

        Returns:
            torch.Tensor: A boolean tensor indicating which category indices in `subset` are less than each label in `y`.

        Raises:
            RuntimeError: If the tensor conversion or computation fails.

        Example:
            ```python
            y = torch.tensor([1, 3])
            subset = [0, 1, 2, 3]
            lbclabel = LBCLabel(y, subset, device='cpu')
            result = lbclabel()
            print(result)
            ```
            Output:
            ```
            tensor([[ True, False, False, False],
                    [ True,  True,  True, False]])
            ```
        """
        logger.info("__call__ starts")
        try:
            # Convert subset to a tensor on the specified device and reshape for comparison
            subset_tensor = torch.tensor(self.subset, device=self.device).view(1, -1)
            # Compare the subset tensor with the label tensor
            result = subset_tensor < self.y.view(-1, 1)
            logger.info("__call__ ends")
            return result
        except Exception as e:
            logger.error(f"Error during label vector conversion: {e}")
            raise RuntimeError("Failed to convert label numbers into label vectors.") from e
