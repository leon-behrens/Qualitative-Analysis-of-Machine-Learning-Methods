import pandas as pd
import os
from transformers.utils import add_start_docstrings_to_model_forward
from datetime import datetime
from src.main.python.neuralNet.MyModel import MyModel
import random
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

from src.main.python.neuralNet.lbcUtils.LBCWithLogitsLoss import LBCWithLogitsLoss
from src.main.python.neuralNet.loops.ConfusionLoop import ConfusionLoop
from src.main.python.neuralNet.loops.TrainLoop import TrainLoop

class Training:
    def __init__(self, n_categories, n_categories_dataset, subset, learning_rate):
        # initialize model
        self.subset = subset
        self.my_model_instance = MyModel(n_categories=n_categories)
        self.model = self.my_model_instance()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.loss_fn = LBCWithLogitsLoss(n_categories_dataset, subset, self.device)

        # define the optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # training and evaluation loop. evaluation here only on training dataset itself for speed.
        self.train_losses = []
        self.errs = []
        self.valid_losses = []

        #evaluation
        self.err = 0
        self.eval_dataset = None
        self.n_categories_dataset = 0
        self.ds_size = 0

        #training
        self.training_dataset = None
        self.record_every = 0
        self.batch_size = 0

    def set_evaluation_parameters(self, eval_dataset, n_categories_dataset, ds_size):
        self.eval_dataset = eval_dataset
        self.n_categories_dataset = n_categories_dataset
        self.ds_size = ds_size

    def set_training_parameters(self, training_dataset, record_every, batch_size):
        self.training_dataset = training_dataset
        self.record_every = record_every
        self.batch_size = batch_size



    def train(self, epochs):
        fig = plt.figure()
        for t in range(epochs):
            print(f"Epoch {t + 1}\n---------------------------------------------------------")

            # validation loop including confusion signal
            self.__evaluation__()

            # plotting
            plt.semilogy(self.subset, self.err, '-d')
            plt.xlabel("system parameter")
            plt.ylabel("validation error")
            plt.show()

            plt.plot(self.valid_losses)
            plt.title("validation loss")
            plt.show()

            # plot train loss
            self.__training__()
            plt.semilogy(self.train_losses)
            plt.xlabel(f'seen samples [{self.record_every * self.batch_size}]')
            plt.ylabel('loss')
            plt.show()

            self.__save_checkpoint__(epoch=t + 1, loss=self.train_losses[-1])

        self.__save_fig__(fig)
        self.__save_model__(self.model, model_name="trained_model")



    def __evaluation__(self):
        conf, valid_loss = ConfusionLoop(self.eval_dataset, self.model,
                                         self.loss_fn, self.n_categories_dataset, self.subset)
        self.valid_losses.append(valid_loss)
        self.err = conf.detach().cpu().numpy() / self.ds_size
        self.errs.append(self.err)


    def __training__(self):
        loss = TrainLoop(self.training_dataset, self.model, self.loss_fn,
                         self.optimizer, self.device, self.subset, self.record_every)
        self.train_losses.append(loss)

    @staticmethod
    def __save_fig__(fig, name="loss_curve"):
        """
        Saves the given matplotlib figure with a timestamp.

        Args:
            fig (matplotlib.figure.Figure): The figure to save.
            name (str): The base name for the figure file.
        """
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Specify the folder path and filename
        plot_folder = "/Users/leon/Uni/Master/Projektarbeit/Projektarbeit/src/main/resources/plots"
        os.makedirs(plot_folder, exist_ok=True)
        plot_path = os.path.join(plot_folder, f"{name}_{timestamp}.png")

        # Save the plot
        fig.savefig(plot_path)
        plt.close(fig)  # Close the plot to free up memory

        print(f"Figure saved to {plot_path}")

    @staticmethod
    def __save_model__(model, model_name="model"):
        """
        Saves the model weights and the full model with a timestamp.

        Args:
            model (torch.nn.Module): The model to save.
            model_name (str): The base name for the model file.
        """
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Specify the folder path
        model_folder = "/Users/leon/Uni/Master/Projektarbeit/Projektarbeit/src/main/resources/models"
        os.makedirs(model_folder, exist_ok=True)

        # Save model weights
        weights_path = os.path.join(model_folder, f"{model_name}_weights_{timestamp}.pth")
        torch.save(model.state_dict(), weights_path)

        # Save the entire model
        full_model_path = os.path.join(model_folder, f"{model_name}_full_{timestamp}.pth")
        torch.save(model, full_model_path)

        print(f"Model weights saved to {weights_path}")
        print(f"Full model saved to {full_model_path}")

    @staticmethod
    def __save_checkpoint__(epoch, loss, file_name="checkpoint_data"):
        """
        Saves or appends checkpoint data to a CSV file with a timestamp-based filename.

        Args:
            epoch (int): The current epoch number.
            loss (float): The loss value.
            file_name (str): The base name for the checkpoint file.
        """
        # Generate timestamp for the file
        timestamp = datetime.now().strftime("%Y%m%d")
        checkpoint_folder = "/Users/leon/Uni/Master/Projektarbeit/Projektarbeit/src/main/resources/checkpoints"
        os.makedirs(checkpoint_folder, exist_ok=True)

        checkpoint_path = os.path.join(checkpoint_folder, f"{file_name}_{timestamp}.csv")

        # Create or append to the CSV
        checkpoint_data = {'epoch': epoch, 'loss': loss}
        df = pd.DataFrame([checkpoint_data])

        if os.path.exists(checkpoint_path):
            df.to_csv(checkpoint_path, mode='a', header=False, index=False)
        else:
            df.to_csv(checkpoint_path, index=False)

        print(f"Checkpoint data saved to {checkpoint_path}")

    # evaluation of results from final loop
    def get_last_loop(self):
        conf, valid_loss = ConfusionLoop(self.eval_dataset, self.model, self.loss_fn,
                                         self.n_categories_dataset, subset=self.subset)
        self.valid_losses.append(valid_loss)
        self.err = conf.detach().cpu().numpy() / self.ds_size
        self.errs.append(self.err)
        return self.errs, self.train_losses, self.valid_losses


