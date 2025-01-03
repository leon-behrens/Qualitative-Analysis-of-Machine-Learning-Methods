import pickle
from IPython.display import clear_output
import random
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

from src.main.python.neuralNet.MyModel import MyModel
from src.main.python.neuralNet.lbcUtils.LBCWithLogitsLoss import LBCWithLogitsLoss
from src.main.python.neuralNet.loops.ConfusionLoop import ConfusionLoop
from src.main.python.neuralNet.loops.TrainLoop import TrainLoop
from src.main.python.neuralNet.CustomDataset import CustomDataset
from src.main.resources.CreateLogger import CreateLogger


create_logger = CreateLogger("MainAnalysis")
logger = create_logger.return_logger()

__pdoc__ = {
    "MainAnalysis": False,   # Exclude this class from documentation
    "__init__": False,  # Exclude this function from documentation
}
class MainAnalysis:
    def __init__(self):
        try:
            logger.info("Initializing starts")
            self.TRAIN_FOLDER = "/Users/leon/Uni/Master/Projektarbeit/Qualitative-Analysis-of-Machine-Learning-Methods/src/main/resources/data/pictures/tech_train"
            self.EVAL_FOLDER = "/Users/leon/Uni/Master/Projektarbeit/Qualitative-Analysis-of-Machine-Learning-Methods/src/main/resources/data/pictures/tech_eval"

            # imagnet parameters. preprocess image sizes for faster training.
            logger.info("transform")
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            logger.info("dataset")
            self.train_dataset = CustomDataset(directory=self.TRAIN_FOLDER, transform = self.transform)
            self.eval_dataset = CustomDataset(directory=self.EVAL_FOLDER, transform = self.transform)

            # set num_workers and persistent_workers for faster dataloaders
            logger.info("dataloader")
            self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=32, shuffle=True)
            self.eval_dataloader = torch.utils.data.DataLoader(self.eval_dataset, batch_size=32, shuffle=True)
            self.n_categories_dataset = 150  # number of categories in dataset

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info("Initializing ends")

        except Exception as e:
            logger.error(e)
            raise e

    def __call__(self):
        logger.info("__call__ starts")
        # define other required imports and function definitions here

        ds_size = len(self.eval_dataset)
        batch_size = 16 * 64

        # define training parameters
        learning_rate = 0.5 * 1e-4
        epochs = 151
        record_every = 10
        subset = list(range(self.n_categories_dataset - 1))

        n_categories = len(subset) + 1

        # initialize model
        logger.info("creating model starts")
        my_model = MyModel(n_categories)
        model = my_model()
        logger.info("creating model starts")

        logger.info("LBCWithLogitsLoss starts")
        loss_fn = LBCWithLogitsLoss(self.n_categories_dataset, subset, device=self.device)

        # define the optimizer
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=learning_rate)

        # training and evaluation loop. evaluation here only on training dataset itself for speed.
        losses = []
        errs = []
        valid_losses = []

        logger.info("training starts")
        for t in range(epochs):
            logger.info(f"Epoch {t + 1}\n----")
            print(f"Epoch {t + 1}\n-------------------------------")

            # validation loop including confusion signal
            logger.info("creating confusion loop starts")
            confusion_loop = ConfusionLoop(self.eval_dataloader, model, loss_fn, self.n_categories_dataset, subset=subset)
            conf, valid_loss = confusion_loop()
            valid_losses.append(valid_loss)
            err = conf.detach().cpu().numpy() / ds_size
            errs.append(err)

            # plotting
            plt.semilogy(subset, err, '-d')
            plt.xlabel('system parameter')
            plt.ylabel('error')
            plt.show()

            plt.plot(valid_losses)
            plt.title('vlosses')
            plt.show()

            # training loop
            train_loop = TrainLoop(self.train_dataloader, model, loss_fn, optimizer, device = self.device, subset=subset)
            loss = train_loop()
            losses += loss

            # plot train loss
            plt.semilogy(losses)
            plt.xlabel(f'seen samples [{record_every * batch_size}]')
            plt.ylabel('loss')
            plt.show()

        # evaluation of results from final loop
        logger.info("evaluating starts")
        confusion_loop = ConfusionLoop(self.eval_dataloader, model, loss_fn, self.n_categories_dataset, subset=subset)
        conf, valid_loss = confusion_loop()
        valid_losses.append(valid_loss)
        err = conf.detach().cpu().numpy() / ds_size
        errs.append(err)

        # save results
        results = {'errs': errs, 'losses': losses, 'valid_losses': valid_losses}



if __name__ == "__main__":
    analysis = MainAnalysis()
    analysis()