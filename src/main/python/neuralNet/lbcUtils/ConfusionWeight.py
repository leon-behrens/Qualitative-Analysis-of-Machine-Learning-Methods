import torch
from main.resources.CreateLogger import CreateLogger

create_logger = CreateLogger("ConfusionWeight")
logger = create_logger.return_logger()

class ConfusionWeight:
    def __init__(self, n_categories_total, subset, device='cuda'):
        """
        Initializes the ConfusionWeight class.

        Args:
            n_categories_total (int): Total number of categories.
            subset (list or torch.Tensor): Subset of categories to calculate weights for.
            device (str): Computation device ('cuda' or 'cpu').
        """
        logger.info("Initialization starts")
        self.n_categories_total = n_categories_total

        # Validate subset
        if subset is None or len(subset) == 0:
            logger.error("Subset is None or empty.")
            raise ValueError("Subset cannot be None or empty.")
        if not all(0 <= idx < self.n_categories_total for idx in subset):
            logger.error("Subset contains invalid indices.")
            raise ValueError("Subset contains invalid indices.")

        self.subset = subset

        # Create and validate device
        try:
            self.device = torch.device(device)
        except Exception as e:
            logger.error(f"Invalid device specified: {device}. Error: {e}")
            raise ValueError(f"Invalid device: {device}") from e

        logger.info("Initialization ends")

    def __call__(self, reshape=True):
        """
        Calculates weights to correct for class imbalance based on the subset of categories.

        Args:
            reshape (bool): Whether to reshape the weights to (1, -1). Default is True.

        Returns:
            torch.Tensor: A tensor of positive weights for the specified subset of categories.
                          The weights are normalized by the total number of categories.

        Raises:
            ValueError: If the subset contains invalid indices or if the tensor creation fails.
        """
        logger.info("__call__ starts")
        try:
            weights = torch.arange(1, self.n_categories_total + 1, device=self.device)[
                          self.subset] / self.n_categories_total
            logger.debug(f"Calculated weights: {weights}")
            logger.info("__call__ ends")
            return weights.view(1, -1) if reshape else weights
        except Exception as e:
            logger.error(f"Error calculating confusion weights: {e}")
            raise ValueError("Failed to calculate confusion weights.") from e
