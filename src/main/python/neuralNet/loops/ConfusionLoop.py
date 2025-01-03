# Import definitions of classes and functions for learning by confusion
from src.main.python.neuralNet.lbcUtils.ConfusionWeight import ConfusionWeight
from src.main.python.neuralNet.lbcUtils.LBCLabel import LBCLabel
import torch
from src.main.python.neuralNet.lbcUtils.LBCWithLogitsLoss import LBCWithLogitsLoss
from src.main.resources.CreateLogger import CreateLogger

# Create a logger instance
create_logger = CreateLogger("ConfusionLoop")
logger = create_logger.return_logger()

class ConfusionLoop:
    """
    A class to perform a confusion-based evaluation loop for a neural network model.

    This class calculates the confusion error and loss over a dataset using a given model,
    loss function, and dataloader. It supports handling class imbalance through weighted loss computation.

    Attributes:
        dataloader (torch.utils.data.DataLoader): The data loader for the evaluation data.
        model (torch.nn.Module): The neural network model to evaluate.
        loss_fn (callable): The loss function to use for evaluation.
        n_categories (int): The total number of categories in the dataset.
        device (str): The computation device ('cuda' or 'cpu').
        subset (list or torch.Tensor): Indices representing the subset of categories to consider.
        torch_weight (torch.Tensor): Weights for correcting class imbalance.
        running_conf (torch.Tensor): Accumulated confusion errors during evaluation.
        running_loss (float): Accumulated loss over the evaluation loop.
    """

    def __init__(self, dataloader, model, loss_fn, n_categories, device="cpu", subset=None):
        """
        Initializes the ConfusionLoop class.

        Args:
            dataloader (torch.utils.data.DataLoader): The data loader for the evaluation data.
            model (torch.nn.Module): The neural network model to evaluate.
            loss_fn (callable): The loss function to use for evaluation.
            n_categories (int): The total number of categories in the dataset.
            device (str, optional): The device to perform computations on ('cuda' or 'cpu'). Default is 'cuda'.
            subset (list or torch.Tensor, optional): A subset of category indices to consider. Default is None.

        Raises:
            RuntimeError: If an error occurs during the initialization of weights.
        """
        logger.info("Initializing ConfusionLoop starts")
        self.dataloader = dataloader
        self.model = model
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
            self.torch_weight = confusion_weight()

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

        Raises:
            RuntimeError: If an error occurs during the evaluation loop.
        """
        logger.info("__call__ starts")



        try:
            with torch.no_grad():
                for X, y in self.dataloader:
                    logger.info(f"Dataloader output - X shape: {X.shape}, y: {y}")
                    X, y = X.to(self.device), y.to(self.device)

                    # Forward pass: get model predictions
                    pred = self.model(X)

                    # Convert predictions and labels to boolean values
                    pred_bool = torch.sigmoid(pred) > 0.5

                    if y is None or len(y) == 0:
                        logger.error("Labels (y) are missing or empty.")
                        raise ValueError("Labels (y) cannot be None or empty.")

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

