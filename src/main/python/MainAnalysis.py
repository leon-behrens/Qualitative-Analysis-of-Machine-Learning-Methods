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

create_logger = CreateLogger("MainAnalysis")
logger = create_logger.return_logger()

__pdoc__ = {
    "MainAnalysis": False,
    "__init__": False,
}


class MainAnalysis:
    def __init__(self):
        try:
            logger.info("Initializing starts")
            self.TRAIN_FOLDER = "/scicore/home/bruder/behleo00/PA/src/main/resources/data/pictures/tech_train"
            self.EVAL_FOLDER = "/scicore/home/bruder/behleo00/PA/src/main/resources/data/pictures/tech_eval"

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
        #epochs = 151
        epochs = 79
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

        # CSV logger for iteration logs
        csv_logger = IterationLogger(
            log_dir="/scicore/home/bruder/behleo00/PA/src/main/resources/data/",
            filename="training_log"
        )

        # Training and evaluation loop
        logger.info("training starts")
        plot_data = {
            "epochs": [],
            "train_losses": [],
            "valid_losses": [],
            "validation_errors": []
        }

        for t in range(epochs):
            logger.info(f"Epoch {t + 1}\n----")
            print(f"Epoch {t + 1}\n-------------------------------")

            # Validation loop
            logger.info("creating confusion loop starts")
            confusion_loop = ConfusionLoop(self.eval_dataloader, model, loss_fn, self.n_categories_dataset, subset=subset)
            conf, valid_loss = confusion_loop()
            err = conf.detach().cpu().numpy() / ds_size

            # Log validation data
            step = 0  # Reset step for validation
            csv_logger.log(epoch=t + 1, step=step, train_loss=None, valid_loss=valid_loss, error=err.mean())
            plot_data["valid_losses"].append(valid_loss)
            plot_data["validation_errors"].append(err.mean())

            # Training loop
            train_loop = TrainLoop(self.train_dataloader, model, loss_fn, optimizer, device=self.device, subset=subset)
            losses = train_loop()  # Returns a list of training losses for each batch

            for batch_idx, train_loss in enumerate(losses):
                step = batch_idx + 1  # Step corresponds to the batch index
                csv_logger.log(epoch=t + 1, step=step, train_loss=train_loss)

            plot_data["train_losses"].extend(losses)
            plot_data["epochs"].append(t + 1)


        # Save results
        self.save_results_individual(plot_data)
        self.save_model(model)

        saver = SaveResults()
        saver.save_each_as_csv(plot_data)


    def save_model(self, model, model_name="trained_model"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = "/scicore/home/bruder/behleo00/PA/src/main/resources/data/"
        os.makedirs(model_dir, exist_ok=True)

        # Save model weights and full model
        weights_path = os.path.join(model_dir, f"{model_name}_weights_{timestamp}.pth")
        torch.save(model.state_dict(), weights_path)

        full_model_path = os.path.join(model_dir, f"{model_name}_full_{timestamp}.pth")
        torch.save(model, full_model_path)

        print(f"Model weights saved to {weights_path}")
        print(f"Full model saved to {full_model_path}")

    def save_results_individual(self, data, result_dir="/scicore/home/bruder/behleo00/PA/src/main/resources/data/"):
        os.makedirs(result_dir, exist_ok=True)

        # Save each key in its own CSV
        for key, values in data.items():
            csv_path = os.path.join(result_dir, f"{key}.csv")
            df = pd.DataFrame({key: values})
            df.to_csv(csv_path, index=False)
            print(f"{key.capitalize()} saved to {csv_path}")


if __name__ == "__main__":
    analysis = MainAnalysis()
    analysis()
