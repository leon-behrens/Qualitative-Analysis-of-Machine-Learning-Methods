import pickle
from IPython.display import clear_output
import random
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import datetime
import pandas as pd
from datetime import datetime

from main.python.neuralNet.MyModel import MyModel
from main.python.neuralNet.lbcUtils.LBCWithLogitsLoss import LBCWithLogitsLoss
from main.python.neuralNet.loops.ConfusionLoop import ConfusionLoop
from main.python.neuralNet.loops.TrainLoop import TrainLoop
from main.python.neuralNet.CustomDataset import CustomDataset
from main.python.results.IterationLogger import IterationLogger
from main.resources.CreateLogger import CreateLogger
from main.python.results.SaveResults import SaveResults

create_logger = CreateLogger("MainAnalysisSD35")
logger = create_logger.return_logger()

class MainAnalysisSD35:
    def __init__(self):
        try:
            logger.info("Initializing starts")
            self.TRAIN_FOLDER = "/scicore/home/bruder/behleo00/PA/src/main/resources/data/pictures/tech_train_SD35"
            self.EVAL_FOLDER = "/scicore/home/bruder/behleo00/PA/src/main/resources/data/pictures/tech_eval_SD35"

            # ImageNet parameters: preprocess image sizes for faster training
            logger.info("transform")
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            logger.info("dataset")
            self.train_dataset = CustomDataset(directory=self.TRAIN_FOLDER, transform=self.transform)
            self.eval_dataset = CustomDataset(directory=self.EVAL_FOLDER, transform=self.transform)

            # Set num_workers and persistent_workers for faster dataloaders
            logger.info("dataloader")
            self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=32, shuffle=True)
            self.eval_dataloader = torch.utils.data.DataLoader(self.eval_dataset, batch_size=32, shuffle=True)
            self.n_categories_dataset = 150  # Number of categories in dataset

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Device: {self.device}")
            logger.info("Initializing ends")

        except Exception as e:
            logger.error(e)
            raise e

    def __call__(self):
        logger.info("__call__ starts")

        ds_size = len(self.eval_dataset)
        batch_size = 16 * 64

        # Define training parameters
        learning_rate = 0.5 * 1e-4
        epochs = 151
        subset = list(range(self.n_categories_dataset - 1))
        n_categories = len(subset) + 1

        # Initialize model
        logger.info("creating model starts")
        my_model = MyModel(n_categories)
        model = my_model()
        logger.info("creating model ends")

        logger.info("LBCWithLogitsLoss starts")
        loss_fn = LBCWithLogitsLoss(self.n_categories_dataset, subset, device=self.device)

        # Define the optimizer
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=learning_rate)

        record_every=10

        # Training and evaluation loop
        logger.info("training starts")
        
        plot_dir = "/scicore/home/bruder/behleo00/PA/src/main/resources/data/results1daySD35/plots"
        os.makedirs(plot_dir, exist_ok=True)

        trainingslosses = []
        validationerrs = []
        valid_losses = []

        for t in range(epochs):
            logger.info(f"Epoch {t+1}\n-------------------------------")
            logger.error(f"Epoch {t+1}\n-------------------------------")

            # validation loop including confusion signal
            confusion_loop = ConfusionLoop(self.eval_dataloader, model, loss_fn, self.n_categories_dataset, subset=subset)
            conf, valid_loss = confusion_loop()
            valid_losses.append(valid_loss)
            err = conf.detach().cpu().numpy() / ds_size
            validationerrs.append(err)

            # plotting
            plt.figure()
            plt.semilogy(subset, err,'-d')
            plt.xlabel('system parameter')
            plt.ylabel('error')
            plt.savefig(os.path.join(plot_dir, f"system-parameter-vs-error_epoch_{t+1}.png"))
            plt.close() 


            plt.figure()
            plt.plot(valid_losses)
            plt.title('vlosses')
            plt.savefig(os.path.join(plot_dir, f"vlosses_epoch_{t+1}.png"))
            plt.close()

            # training loop
            train_loop = TrainLoop(self.train_dataloader, model, loss_fn, optimizer, device=self.device, subset=subset)
            loss = train_loop()
            trainingslosses += loss

            # plot train loss
            plt.figure()
            plt.semilogy(trainingslosses)
            plt.xlabel(f'seen samples [{record_every * batch_size}]')
            plt.ylabel('loss')
            plt.savefig(os.path.join(plot_dir, f"trainings-loss_epoch_{t+1}.png"))
            plt.close()

            logger.info(f"Confusion matrix: {conf}, Validation loss: {valid_loss}")
            logger.info(f"Error: {err}, All errors: {validationerrs}")
            logger.info(f"Loss: {loss}, All losses: {trainingslosses}")


        # evaluation of results from final loop
        confusion_loop2 = ConfusionLoop(self.eval_dataloader, model, loss_fn, self.n_categories_dataset, subset=subset)
        conf, valid_loss = confusion_loop2()
        valid_losses.append(valid_loss)
        err = conf.detach().cpu().numpy() / ds_size
        validationerrs.append(err)

        # save results
        resultsSD35 = {'errs': validationerrs, 'losses': trainingslosses, 'valid_losses': valid_losses}
        self.save_model(model)

        saver = SaveResults()
        saver.save_as_csv(resultsSD35)
        saver.save_each_as_csv(resultsSD35)


    def save_model(self, model, model_name="trained_model"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = "/scicore/home/bruder/behleo00/PA/src/main/resources/data/results1daySD35/"
        os.makedirs(model_dir, exist_ok=True)

        # Save model weights and full model
        weights_path = os.path.join(model_dir, f"{model_name}_weights_{timestamp}.pth")
        torch.save(model.state_dict(), weights_path)

        full_model_path = os.path.join(model_dir, f"{model_name}_full_{timestamp}.pth")
        torch.save(model, full_model_path)

        print(f"Model weights saved to {weights_path}")
        print(f"Full model saved to {full_model_path}")


if __name__ == "__main__":
    analysis = MainAnalysisSD35()
    analysis()
