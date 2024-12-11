# Import definitions of classes and functions for learning by confusion
from src.main.python.neuralNet.lbcUtils.LBCLabel import LBCLabel
import torch

from src.main.python.neuralNet.lbcUtils.LBCWithLogitsLoss import LBCWithLogitsLoss
from src.main.resources.CreateLogger import CreateLogger

# Create a logger instance
create_logger = CreateLogger("ConfusionLoop")
logger = create_logger.return_logger()

class TrainLoop:
    """
    A class to perform a training loop for learning by confusion (LBC).

    This class calculates the confusion error and loss during training, helping to
    assess model performance when dealing with class imbalances.

    Attributes:
        dataloader (torch.utils.data.DataLoader): The data loader for the training data.
        model (torch.nn.Module): The neural network model being trained.
        loss_fn (callable): The loss function to use for training.
        n_categories (int): The total number of categories in the dataset.
        device (str): The computation device ('cuda' or 'cpu').
        subset (list or torch.Tensor): Indices representing the subset of categories to consider.
        torch_weight (torch.Tensor): Weights for correcting class imbalance.
        running_conf (torch.Tensor): Accumulated confusion errors.
        running_loss (float): Accumulated loss over the training loop.
    """

    def __init__(self, dataloader, model, loss_fn, n_categories, device='cuda', subset=None):
        """
        Initializes the TrainLoop class.

        Args:
            dataloader (torch.utils.data.DataLoader): The data loader for the training data.
            model (torch.nn.Module): The neural network model to train.
            loss_fn (callable): The loss function to use for training.
            n_categories (int): The total number of categories in the dataset.
            device (str, optional): The device to perform computations on ('cuda' or 'cpu'). Default is 'cuda'.
            subset (list or torch.Tensor, optional): A subset of category indices to consider. Default is None.

        Raises:
            RuntimeError: If an error occurs during the initialization of weights.
        """
        logger.info("Initializing TrainLoop starts")
        self.dataloader = dataloader
        self.model = model
        self.loss_fn = loss_fn
        self.n_categories = n_categories
        self.device = device
        self.subset = subset

        try:
            self.torch_weight = LBCWithLogitsLoss.confusion_weight(
                self.n_categories, self.subset, device=self.device
            ).view(1, -1)
        except Exception as e:
            logger.error(f"Error initializing confusion weights: {e}")
            raise RuntimeError("Failed to initialize confusion weights.") from e

        self.running_conf = torch.zeros(self.n_categories - 1, device=self.device)
        self.running_loss = 0
        logger.info("Initializing TrainLoop ends")

    def __call__(self):
        """
        Executes the training loop and computes the confusion error and total loss.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The running confusion error, scaled by 0.5.
                - float: The total running loss.

        Raises:
            RuntimeError: If an error occurs during the training loop.

        Example:
            ```python
            train_loop = TrainLoop(dataloader, model, loss_fn, n_categories=10, device='cpu')
            running_conf, running_loss = train_loop()
            print("Confusion Error:", running_conf)
            print("Total Loss:", running_loss)
            ```
        """
        logger.info("Training loop starts")
        try:
            with torch.no_grad():
                for X, y in self.dataloader:
                    X, y = X.to(self.device), y.to(self.device)

                    # Model prediction
                    pred = self.model(X)

                    # Convert predictions and labels to boolean values
                    pred_bool = torch.sigmoid(pred) > 0.5
                    Y_bool = LBCLabel(y, self.subset)
                    Y = Y_bool.float()

                    # Calculate confusion error
                    confusion = (
                        1. / (1. - self.torch_weight) * (pred_bool != Y_bool) * (Y_bool == 1) +
                        1. / (self.torch_weight) * (pred_bool != Y_bool) * (Y_bool == 0)
                    ).sum(0)

                    # Update running confusion and loss
                    self.running_conf += confusion
                    loss = self.loss_fn(pred, Y)
                    self.running_loss += loss.item()
            logger.info("Training loop ends")
            return 0.5 * self.running_conf, self.running_loss

        except Exception as e:
            logger.error(f"Error during training loop: {e}")
            raise RuntimeError("Training loop failed.") from e
