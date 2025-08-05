import torch
from main.python.neuralNet.lbcUtils.ConfusionWeight import ConfusionWeight
from main.resources.CreateLogger import CreateLogger

create_logger = CreateLogger("LBCWithLogitsLoss")
logger = create_logger.return_logger()

class LBCWithLogitsLoss:
    """
    A custom loss function for unbiased, multi-task learning by confusion (LBC).

    This loss function applies a weighted binary cross-entropy loss to correct for class imbalance.
    """

    def __init__(self, n_categories_total, subset, device="cuda"):
        """
        Initializes the LBCWithLogitsLoss class.

        Args:
            n_categories_total (int): Total number of categories in the dataset.
            subset (list or torch.Tensor): Subset of categories to calculate the loss for.
            device (str, optional): Device for computations ('cuda' or 'cpu'). Default is 'cuda'.

        Raises:
            ValueError: If initialization fails due to invalid inputs.
        """
        logger.info("Initializing LBCWithLogitsLoss starts")
        self.device = torch.device(device)  # Ensure valid device
        self.n_categories_total = n_categories_total
        self.subset = subset

        # Initialize confusion weight
        try:
            confusion_weight = ConfusionWeight(self.n_categories_total, self.subset, self.device)
            self.pos_weight = confusion_weight().to(self.device)  # Ensure pos_weight is on the correct device
            self.logsig = torch.nn.LogSigmoid()
            logger.info("Initializing LBCWithLogitsLoss ends")
        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            raise ValueError("Failed to initialize LBCWithLogitsLoss.") from e

    def __call__(self, logits, targets):
        logger.info("__call__ starts")

        # Validate input tensors
        if not isinstance(logits, torch.Tensor) or not isinstance(targets, torch.Tensor):
            logger.error("Logits and targets must be torch.Tensor.")
            raise ValueError("Logits and targets must be torch.Tensor.")
        if logits.shape != targets.shape:
            logger.error(f"Shape mismatch: logits {logits.shape}, targets {targets.shape}")
            raise ValueError("Logits and targets must have the same shape.")
        if not torch.all((targets == 0) | (targets == 1)):
            logger.error("Targets contain non-binary values.")
            raise ValueError("Targets must contain only binary values (0 or 1).")

        # Ensure tensors are on the correct device
        logits = logits.to(self.device)
        targets = targets.to(self.device)

        # Convert targets to float to allow arithmetic operations
        targets = targets.float()

        try:
            # Compute the weighted binary cross-entropy loss
            loss = - (targets * self.logsig(logits) / (1. - self.pos_weight)
                    + (1. - targets) * self.logsig(-logits) / self.pos_weight)

            logger.debug(f"Computed loss tensor: {loss}")
            mean_loss = torch.mean(loss)
            logger.info("__call__ ends")
            return mean_loss
        except Exception as e:
            logger.error(f"Error during loss computation: {e}")
            raise RuntimeError("Failed to compute the loss.") from e


