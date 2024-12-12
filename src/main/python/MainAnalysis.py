# import definitions of classes and functions for learning by confusion
from neuralNet.lbcUtils import *
import pickle
from IPython.display import clear_output
import random
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

from src.main.python.neuralNet.CustomDataset import CustomDataset
from src.main.resources.CreateLogger import CreateLogger
import pdoc


create_logger = CreateLogger("MainAnalysis")
logger = create_logger.return_logger()

__pdoc__ = {
    "MainAnalysis": False,   # Exclude this class from documentation
    "__init__": False,  # Exclude this function from documentation
}
class MainAnalysis:
    def __init__(self):
        self.TRAIN_FOLDER = "/Users/leon/Uni/Master/Projektarbeit/Projektarbeit/src/main/resources/data/pictures/tech_train"
        self.EVAL_FOLDER = "/Users/leon/Uni/Master/Projektarbeit/Projektarbeit/src/main/resources/data/pictures/tech_eval"

        # imagnet parameters. preprocess image sizes for faster training.
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.train_dataset = CustomDataset(directory=self.TRAIN_FOLDER, transform = self.transform)
        self.eval_dataset = CustomDataset(directory=self.EVAL_FOLDER, transform = self.transform)

        # set num_workers and persistent_workers for faster dataloaders
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=32, shuffle=True)
        self.eval_loader = torch.utils.data.DataLoader(dataset=self.eval_dataset, batch_size=32, shuffle=True)

        # number of categories in dataset
        self.n_categories_dataset = 150

        # define other required imports and function definitions here
        self.ds_size = len(self.eval_dataset)
        self.batch_size = 16*64

        # define training parameters
        self.learning_rate = 0.5 * 1e-4
        self.epochs = 151
        self.record_every = 10
        self.subset = list[range(self.n_categories_dataset -1)]
        self.n_categories = len(self.subset) + 1

