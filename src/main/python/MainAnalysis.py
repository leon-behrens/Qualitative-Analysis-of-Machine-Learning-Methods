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


from src.main.python.neuralNet.MyModel import MyModel
from src.main.python.neuralNet.lbcUtils.LBCWithLogitsLoss import LBCWithLogitsLoss
from src.main.python.neuralNet.loops.ConfusionLoop import ConfusionLoop
from src.main.python.neuralNet.loops.TrainLoop import TrainLoop
from src.main.python.neuralNet.CustomDataset import CustomDataset
from src.main.python.results.IterationLogger import IterationLogger
from src.main.python.results.SaveResults import SaveResults
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
        #epochs = 151
        epochs = 3
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

        csv_logger = IterationLogger(
            log_dir="/Users/leon/Uni/Master/Projektarbeit/Qualitative-Analysis-of-Machine-Learning-Methods/src/main/resources/data/",
            filename="training_log")


        # training and evaluation loop. evaluation here only on training dataset itself for speed.
        losses = []
        errs = []
        valid_losses = []

        # Initialize persistent plots
        plt.ion()
        fig, (ax_train, ax_valid, ax_error) = plt.subplots(3, 1, figsize=(10, 15))
        ax_train.set_title("Training Loss Over Time")
        ax_train.set_xlabel("Seen Samples")
        ax_train.set_ylabel("Loss")

        ax_valid.set_title("Validation Loss Over Time")
        ax_valid.set_xlabel("Epoch")
        ax_valid.set_ylabel("Validation Loss")

        ax_error.set_title("Validation Error Over Time")
        ax_error.set_xlabel("Epoch")
        ax_error.set_ylabel("Error")

        # Collect data for saving
        plot_data = {
            "epochs": [],
            "train_losses": [],
            "valid_losses": [],
            "validation_errors": []
        }

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

            # Log validation data
            csv_logger.log(epoch=t + 1, step="Validation", valid_loss=valid_loss, error=err.mean())
            plot_data["valid_losses"].append(valid_loss)
            plot_data["validation_errors"].append(err.mean())

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
            train_losses = loss

            # Log training data
            for step, train_loss in enumerate(train_losses):
                csv_logger.log(epoch=t + 1, step=step + 1, train_loss=train_loss)
            plot_data["train_losses"].extend(train_losses)
            plot_data["epochs"].append(t + 1)

            # Update persistent plots
            ax_train.plot(losses, label=f"Epoch {t + 1}")
            ax_valid.plot(range(1, t + 2), valid_losses, marker='o')
            ax_error.plot(range(1, t + 2), [err.mean() for err in errs], marker='d')

            plt.pause(0.1)

            # plot train loss
            plt.semilogy(losses)
            plt.xlabel(f'seen samples [{record_every * batch_size}]')
            plt.ylabel('loss')
            plt.show()

        # Finalize plots
        plt.ioff()
        plt.savefig("/Users/leon/Uni/Master/Projektarbeit/Qualitative-Analysis-of-Machine-Learning-Methods/src/main/resources/data/training_results.png")
        plt.show()

        # Save data
        self.save_results(plot_data)
        self.save_model(model)


        # evaluation of results from final loop
        logger.info("evaluating starts")
        confusion_loop = ConfusionLoop(self.eval_dataloader, model, loss_fn, self.n_categories_dataset, subset=subset)
        conf, valid_loss = confusion_loop()
        valid_losses.append(valid_loss)
        err = conf.detach().cpu().numpy() / ds_size
        errs.append(err)

        # save results
        save_results = SaveResults()
        results = {
            'errs': errs,
            'losses': losses,
            'valid_losses': valid_losses
        }
        save_results.save_as_csv(results)

    def save_model(self, model, model_name="trained_model"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = "/Users/leon/Uni/Master/Projektarbeit/Qualitative-Analysis-of-Machine-Learning-Methods/src/main/resources/data/"

        os.makedirs(model_dir, exist_ok=True)

        weights_path = os.path.join(model_dir, f"{model_name}_weights_{timestamp}.pth")
        torch.save(model.state_dict(), weights_path)

        full_model_path = os.path.join(model_dir, f"{model_name}_full_{timestamp}.pth")
        torch.save(model, full_model_path)

        print(f"Model weights saved to {weights_path}")
        print(f"Full model saved to {full_model_path}")

    def save_results(self, data, csv_filename="plot_data.csv", pickle_filename="plot_data.pkl"):
        result_dir = "/Users/leon/Uni/Master/Projektarbeit/Qualitative-Analysis-of-Machine-Learning-Methods/src/main/resources/data/"
        os.makedirs(result_dir, exist_ok=True)

        # Save as CSV
        csv_path = os.path.join(result_dir, csv_filename)
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        print(f"Results saved as CSV to {csv_path}")

        # Save as Pickle
        pickle_path = os.path.join(result_dir, pickle_filename)
        with open(pickle_path, "wb") as f:
            pickle.dump(data, f)
        print(f"Results saved as Pickle to {pickle_path}")


if __name__ == "__main__":
    analysis = MainAnalysis()
    analysis()