# Import definitions of classes and functions for learning by confusion
from main.python.neuralNet.lbcUtils.ConfusionWeight import ConfusionWeight
from main.python.neuralNet.lbcUtils.LBCLabel import LBCLabel
import torch
from main.python.neuralNet.lbcUtils.LBCWithLogitsLoss import LBCWithLogitsLoss
from main.resources.CreateLogger import CreateLogger

# Create a logger instance
create_logger = CreateLogger("ConfusionLoop")
logger = create_logger.return_logger()

class ConfusionLoop:
    def __init__(self, dataloader, model, loss_fn, n_categories, device="cuda", subset=None):
        """
        Initializes the ConfusionLoop class.
        """
        logger.info("Initializing ConfusionLoop starts")
        self.dataloader = dataloader
        self.model = model.to(device)  # Move model to the correct device
        self.model.eval()  # Set the model to evaluation mode
        self.loss_fn = loss_fn
        self.n_categories = n_categories
        self.device = device
        self.subset = subset
        self.lbc_label = LBCLabel(subset=self.subset, device=self.device)

        try:
            confusion_weight = ConfusionWeight(
                n_categories_total=self.n_categories,
                subset=self.subset,
                device=self.device
            )
            self.torch_weight = confusion_weight().to(device)  # Ensure weights are on the correct device

            logger.info("Initialized ConfusionLoop's try block successfully")
        except Exception as e:
            logger.error(f"Error initializing confusion weights: {e}")
            raise RuntimeError("Failed to initialize confusion weights.") from e

        self.running_conf = torch.zeros(self.n_categories - 1, device=self.device)
        self.running_loss = 0
        logger.info("Initializing ConfusionLoop ends")

    def __call__(self):
        """
        Executes the evaluation loop and computes the confusion error and total loss.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The running confusion error, scaled by 0.5.
                - float: The total running loss.
        """
        logger.info("__call__ starts")
        self.running_conf = torch.zeros(self.n_categories - 1, device=self.device)  # Reset running confusion
        self.running_loss = 0  # Reset running loss

        try:
            with torch.no_grad():  # Disable gradient tracking
                for X, y in self.dataloader:
                    if X is None or len(X) == 0 or y is None or len(y) == 0:
                        logger.warning("Skipping empty batch.")
                        continue

                    logger.info(f"Dataloader output - X shape: {X.shape}, y shape: {y.shape}")
                    X, y = X.to(self.device), y.to(self.device)

                    # Forward pass: get model predictions
                    pred = self.model(X)

                    # Convert predictions and labels to boolean values
                    pred_bool = torch.sigmoid(pred) > 0.5
                    Y_bool = self.lbc_label(y)  # Pass 'y' to LBCLabel
                    Y = Y_bool.float()

                    # Calculate confusion error
                    confusion = (
                        1. / (1. - self.torch_weight) * (pred_bool != Y_bool) * (Y_bool == 1) +
                        1. / self.torch_weight * (pred_bool != Y_bool) * (Y_bool == 0)
                    ).sum(0)

                    # Update running confusion and loss
                    self.running_conf += confusion
                    loss = self.loss_fn(pred, Y)
                    self.running_loss += loss.item()

            logger.info("__call__ ends")
            return 0.5 * self.running_conf, self.running_loss

        except Exception as e:
            logger.error(f"Error during evaluation loop: {e}")
            raise RuntimeError("Evaluation loop failed.") from e
