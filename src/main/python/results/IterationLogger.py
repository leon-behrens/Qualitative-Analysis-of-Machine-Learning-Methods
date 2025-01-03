import csv
import os
from datetime import datetime

class IterationLogger:
    """
    A utility class to log training and validation results to a CSV file.
    """
    def __init__(self, log_dir, filename="training_log"):
        """
        Initializes the CSVLogger.

        Args:
            log_dir (str): Directory where the log file will be saved.
            filename (str): Base name of the log file (default is "training_log").
        """
        self.log_dir = log_dir
        self.filename = f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.filepath = os.path.join(log_dir, self.filename)
        os.makedirs(log_dir, exist_ok=True)

        # Create the file and write the header
        with open(self.filepath, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Step", "Train Loss", "Validation Loss", "Error"])

    def log(self, epoch, step, train_loss=None, valid_loss=None, error=None):
        """
        Logs a single row of data to the CSV file.

        Args:
            epoch (int): The current epoch number.
            step (int): The current step or batch number.
            train_loss (float, optional): Training loss for this step.
            valid_loss (float, optional): Validation loss for this step.
            error (float, optional): Validation error for this step.
        """
        with open(self.filepath, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([epoch, step, train_loss, valid_loss, error])
