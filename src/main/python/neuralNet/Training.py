from src.main.python.neuralNet.MyModel import MyModel
import random
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

from src.main.python.neuralNet.lbcUtils.LBCWithLogitsLoss import LBCWithLogitsLoss
from src.main.python.neuralNet.loops.ConfusionLoop import ConfusionLoop
from src.main.python.neuralNet.loops.TrainLoop import TrainLoop


class Training():
    def __init__(self, n_categories, n_categories_dataset, subset, learning_rate):
        # initialize model
        self.my_model_instance = MyModel(n_categories=n_categories)
        self.model = self.my_model_instance()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.criterion = LBCWithLogitsLoss(n_categories_dataset, subset, self.device)

        # define the optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # training and evaluation loop. evaluation here only on training dataset itself for speed.
        self.train_losses = []
        self.errs = []
        self.valid_losses = []

        #evaluation
        self.eval_dataset
        self.n_categories_dataset = 0
        self.ds_size = 0

        #training
        self.training_dataset
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



    def train(self, epochs, subset):

        for t in range(epochs):
            print(f"Epoch {t + 1}\n---------------------------------------------------------")

            # validation loop including confusion signal
            err, valid_loss = self.__evaluation__()

            # plotting
            plt.semilogy(subset, err, '-d')
            plt.xlabel("system parameter")
            plt.ylabel("validation error")
            plt.show()

            plt.plot(valid_loss)
            plt.title("validation loss")
            plt.show()

            # plot train loss
            plt.semilogy(self.train_losses)
            plt.xlabel(f'seen samples [{self.record_every * self.batch_size}]')
            plt.ylabel('loss')
            plt.show()


    def __evaluation__(self):
        conf, valid_loss = ConfusionLoop(self.eval_dataset, self.model,
                                         self.criterion, self.n_categories_dataset, subset=self.subset)
        self.valid_losses.append(valid_loss)
        err = conf.detach().cpu().numpy() / self.ds_size
        self.errs.append(err)
        return err, valid_loss

    def __training__(self):
        loss = TrainLoop(self.training_dataset, self.model, self.criterion, self.optimizer, subset = self.subset)
        return self.train_losses.append(loss)
