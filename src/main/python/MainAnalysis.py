import pickle
from IPython.display import clear_output
import random
import torch
from fsspec.asyn import private
from torchvision import transforms
import matplotlib.pyplot as plt

from src.main.python.neuralNet.CustomDataset import CustomDataset
from src.main.python.neuralNet.Training import Training
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
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=4,  # Adjust based on your system's cores
            persistent_workers=True  # Only works if num_workers > 0
        )

        self.eval_loader = torch.utils.data.DataLoader(
            self.eval_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            persistent_workers=True
        )

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

        # start training and evaluation
        self.__training__()




    def __training__(self):
        # start training and evaluation
        self.training = Training(self.n_categories, self.n_categories_dataset, self.subset, self.learning_rate)
        self.training.set_evaluation_parameters(self.eval_dataset, self.n_categories_dataset, self.ds_size)
        self.training.set_training_parameters(self.train_dataset, self.record_every, self.batch_size)
        self.training.train(self.epochs)




if __name__ == "__main__":
    analysis = MainAnalysis()
    for images, targets in analysis.train_loader:
        print(images.shape)  # e.g., torch.Size([32, 3, 224, 224])
        print(targets)