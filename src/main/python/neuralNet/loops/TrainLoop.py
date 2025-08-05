import torch
import time
from main.python.neuralNet.lbcUtils.LBCLabel import LBCLabel
from main.resources.CreateLogger import CreateLogger

# Create a logger instance
create_logger = CreateLogger("TrainLoop")
logger = create_logger.return_logger()

class TrainLoop:
    """
    A class to perform a training loop for learning by confusion (LBC).

    This class calculates the confusion error and loss during training, helping to
    assess model performance when dealing with class imbalances.
    """

    def __init__(self, dataloader, model, loss_fn, optimizer, device="cuda", subset=None, record_every=10):
        """
        Initializes the TrainLoop class.

        Args:
            dataloader (torch.utils.data.DataLoader): The data loader for the training data.
            model (torch.nn.Module): The neural network model to train.
            loss_fn (callable): The loss function to use for training.
            optimizer (torch.optim.Optimizer): The optimizer for training.
            device (str, optional): The device to perform computations on ('cuda' or 'cpu'). Default is 'cuda'.
            subset (list or torch.Tensor, optional): A subset of category indices to consider. Default is None.
            record_every (int, optional): Interval at which to record losses. Default is 10.
        """
        logger.info("Initializing TrainLoop starts")
        self.dataloader = dataloader
        self.model = model.to(device)  # Move model to the correct device
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = torch.device(device)  # Validate device
        self.subset = subset
        self.record_every = record_every
        self.losses = []  # To store batch-level losses
        self.running_conf = 0  # To accumulate confusion errors
        self.size = len(self.dataloader.dataset)
        self.lbc_label = LBCLabel(subset=self.subset, device=self.device)
        logger.info("Initializing TrainLoop ends")

    def __call__(self):
        """
        Executes the training loop and computes the confusion error and total loss.

        Returns:
            dict: A dictionary containing:
                - "losses": List of recorded batch losses.
                - "confusion_error": Total accumulated confusion error across batches.
                - "average_loss": Mean loss across all batches.

        Example:
            ```python
            train_loop = TrainLoop(dataloader, model, loss_fn, optimizer, device='cpu', record_every=5)
            results = train_loop()
            print("Confusion Error:", results["confusion_error"])
            print("Average Loss:", results["average_loss"])
            ```
        """
        logger.info("Training loop starts")
        self.model.train()  # Set model to training mode
        total_loss = 0
        total_batches = len(self.dataloader)

        try:
            for batch, (X, y) in enumerate(self.dataloader):
                start_time = time.time()

                # Skip empty batches
                if X is None or len(X) == 0 or y is None or len(y) == 0:
                    logger.warning(f"Skipping empty batch {batch}.")
                    continue

                # Move data to the correct device
                X, y = X.to(self.device), y.to(self.device)

                # Forward pass
                pred = self.model(X)
                Y = self.lbc_label(y)
                loss = self.loss_fn(pred, Y)

                # Backward pass and optimizer step
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Accumulate total loss
                batch_loss = loss.item()
                total_loss += batch_loss

                # Log batch processing time
                end_time = time.time()
                logger.info(f"Batch {batch}: Loss = {batch_loss:.4f}, Time taken = {end_time - start_time:.2f} seconds")

                # Record loss at intervals
                if batch % self.record_every == 0:
                    self.losses.append(batch_loss)

            # Calculate average loss
            average_loss = total_loss / total_batches
            logger.info("Training loop ends")
            return {
                "losses": self.losses,
                "average_loss": average_loss
            }

        except Exception as e:
            logger.error(f"Error during training loop: {e}")
            raise RuntimeError("Training loop failed.") from e
