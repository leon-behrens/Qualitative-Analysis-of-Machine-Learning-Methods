# import definitions of classes and functions for learning by confusion
from neuralNet.lbcUtils import *
import pickle
from IPython.display import clear_output
import random
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
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
