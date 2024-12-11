import torch
from src.main.resources.CreateLogger import CreateLogger

create_logger = CreateLogger("SimpleCNN")
logger = create_logger.return_logger()

class LBCWithLogitsLoss:
    """
    A custom loss function for unbiased, multi-task learning by confusion (LBC).

    This loss function is designed to correct for class imbalance by applying a weighted
    binary cross-entropy loss. It uses a positive weight (`pos_weight`) calculated based
    on the subset of categories provided, helping to mitigate bias in the learning process.

    Attributes:
        n_categories_total (int): Total number of categories in the dataset.
        subset (list or torch.Tensor): Indices of the categories to calculate the loss for.
        device (str): The computation device ('cuda' or 'cpu').
        pos_weight (torch.Tensor): A tensor of weights for handling class imbalance.
        logsig (torch.nn.LogSigmoid): The LogSigmoid function used for computing the loss.
    """

    def __init__(self, n_categories_total, subset, device='cuda'):
        """
        Initializes the LBCWithLogitsLoss class.

        Args:
            n_categories_total (int):
                The total number of categories in the dataset. This is the range from which categories are sampled.
            subset (list or torch.Tensor):
                A list or tensor containing the indices of the categories for which the loss is calculated.
            device (str, optional):
                The device on which to perform computations ('cuda' or 'cpu'). Default is 'cuda'.

        Attributes Initialized:
            - pos_weight: Calculated using the `confusion_weight` method to adjust for class imbalance.
            - logsig: An instance of the `torch.nn.LogSigmoid` function for computing the log-sigmoid activation.
        """
        logger.info("Initializing starts")
        self.n_categories_total = n_categories_total
        self.subset = subset
        self.device = device

        try:
            self.pos_weight = self.confusion_weight()
            self.logsig = torch.nn.LogSigmoid()
            logger.info("Initializing ends")
        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            raise RuntimeError("Failed to initialize LBCWithLogitsLoss.") from e

    def confusion_weight(self):
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
        logger.info("confusion_weight starts")
        try:
            weights = torch.arange(1, self.n_categories_total + 1,
                                   device=self.device)[self.subset] / self.n_categories_total
            logger.info("confusion_weight ends")
            return weights
        except Exception as e:
            logger.error(f"Error calculating confusion weights: {e}")
            raise ValueError("Failed to calculate confusion weights.") from e

    def __call__(self, logits, targets):
        """
        Computes the custom LBC loss for the given logits and targets.

        The loss is a weighted binary cross-entropy loss, where the weights are used to
        correct for class imbalance. Positive and negative classes are weighted separately
        to ensure balanced learning.

        Args:
            logits (torch.Tensor):
                The raw output logits from the model, with shape `(batch_size, n_classes)`.
            targets (torch.Tensor):
                The ground truth labels, with the same shape as `logits`. Values should be 0 or 1.

        Returns:
            torch.Tensor: The mean loss value across the batch.

        Raises:
            RuntimeError: If an error occurs during the computation of the loss.

        Formula:
            loss = - (targets * log_sigmoid(logits) / (1 - pos_weight)
                      + (1 - targets) * log_sigmoid(-logits) / pos_weight)

        Example:
            ```python
            logits = torch.tensor([[1.2, -0.8, 0.3]], dtype=torch.float32)
            targets = torch.tensor([[1, 0, 1]], dtype=torch.float32)
            loss_fn = LBCWithLogitsLoss(n_categories_total=10, subset=[0, 1, 2], device='cpu')
            loss = loss_fn(logits, targets)
            print(loss)
            ```
        """
        logger.info("__call__ starts")
        try:
            loss = - (targets * self.logsig(logits) / (1. - self.pos_weight)
                      + (1. - targets) * self.logsig(-logits) / self.pos_weight)
            logger.info("__call__ ends")
            return torch.mean(loss)
        except Exception as e:
            logger.error(f"Error during loss computation: {e}")
            raise RuntimeError("Failed to compute the loss.") from e
