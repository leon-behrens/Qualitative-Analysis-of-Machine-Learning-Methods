import torch
from src.main.resources.CreateLogger import CreateLogger

create_logger = CreateLogger("ConfusionWeight")
logger = create_logger.return_logger()

class ConfusionWeight:
    def __init__(self, n_categories_total, subset, device = 'cuda'):
        logger.info("initialization starts")
        self.n_categories_total = n_categories_total
        self.subset = subset
        self.device = device
        logger.info("initialization ends")


    def __call__(self):
        """
        Calculates weights to correct for class imbalance based on the subset of categories.

        The weights are computed as a fraction of the total number of categories. The weight
        for each category is inversely proportional to the number of samples in that category.

        Returns:
            torch.Tensor: A tensor of positive weights for the specified subset of categories.
                          The weights are normalized by the total number of categories.

        Raises:
            ValueError: If the subset contains invalid indices or if the tensor creation fails.

        Example:
            If `n_categories_total = 10` and `subset = [0, 2, 4]`, the output will be:
            tensor([0.1, 0.3, 0.5], device='cuda')
        """
        logger.info("__call__ starts")
        try:
            weights = torch.arange(1, self.n_categories_total + 1, device=torch.device(self.device))[
                          self.subset] / self.n_categories_total
            logger.info("__call__ ends")
            return weights.view(1, -1)
        except Exception as e:
            logger.error(f"Error calculating confusion weights: {e}")
            raise ValueError("Failed to calculate confusion weights.") from e