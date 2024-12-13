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

    def __init__(self, dataset, model, loss_fn, optimizer, device, subset, record_every):
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
        self.dataset = dataset
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.subset = subset
        self.losses = []
        self.size = len(self.dataset)
        self.record_every = record_every


    def __call__(self):
        """
        Executes the training loop and computes the confusion error and total loss.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The running confusion error, scaled by 0.5.
                - float: The total running loss.



        Example:
            ```python
            train_loop = TrainLoop(dataloader, model, loss_fn, n_categories=10, device='cpu')
            running_conf, running_loss = train_loop()
            print("Confusion Error:", running_conf)
            print("Total Loss:", running_loss)
            ```
        """
        logger.info("Training loop starts")
        for batch, (X, y) in enumerate(self.dataset):
            X, y = X.to(self.device), y.to(self.device)
            pred = self.model(X)
            Y = LBCLabel(y, self.subset, self.device).float()
            loss = self.loss_fn(pred, Y)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if batch % self.record_every == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                self.losses.append(loss)
        return self.losses
