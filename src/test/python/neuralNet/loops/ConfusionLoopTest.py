import torch
from torch.utils.data import DataLoader, TensorDataset
from src.main.python.neuralNet.lbcUtils.LBCLabel import LBCLabel
from src.main.python.neuralNet.lbcUtils.LBCWithLogitsLoss import LBCWithLogitsLoss
from src.main.python.neuralNet.lbcUtils.ConfusionWeight import ConfusionWeight
from src.main.python.neuralNet.loops.ConfusionLoop import ConfusionLoop
from src.main.python.neuralNet.SimpleCNN import SimpleCNN
from src.main.resources.CreateLogger import CreateLogger

# Initialize the logger
create_logger = CreateLogger("ConfusionLoopTest")
logger = create_logger.return_logger()

class ConfusionLoopTest:
    def __init__(self):
        # Test parameters
        self.n_categories = 10  # Number of categories
        self.subset = list(range(self.n_categories-1))  # Subset of categories
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # Compute device

        # Mock dataset
        self.batch_size = 8
        self.img_dim = 60  # Image dimension
        X = torch.randn(32, 1, self.img_dim, self.img_dim)  # 32 mock images
        y = torch.randint(0, self.n_categories, (32,))  # Mock labels
        self.dataset = TensorDataset(X, y)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        # Model
        self.model = SimpleCNN(img_dim=self.img_dim, n_categories=self.n_categories).to(self.device)

        # Loss function
        self.loss_fn = LBCWithLogitsLoss(
            n_categories_total=self.n_categories,
            subset=self.subset,
            device=self.device
        )

        # Instantiate ConfusionLoop
        self.confusion_loop = ConfusionLoop(
            dataloader=self.dataloader,
            model=self.model,
            loss_fn=self.loss_fn,
            n_categories=self.n_categories,
            device=self.device,
            subset=self.subset
        )

    def run_test(self):
        try:
            # Run the confusion loop
            print("Running ConfusionLoop...")
            running_conf, running_loss = self.confusion_loop()

            # Validate results
            print(f"Running Confusion Error: {running_conf}")
            print(f"Running Loss: {running_loss}")
            assert running_conf.shape == (self.n_categories - 1,), \
                f"Unexpected confusion error shape: {running_conf.shape}. Expected: ({self.n_categories - 1},)"
            assert isinstance(running_loss, float), \
                f"Running loss should be a float, got {type(running_loss)}."

            print("ConfusionLoop ran successfully!")
            return True

        except Exception as e:
            print(f"Error during testing: {e}")
            logger.error(f"Test failed: {e}")
            return False

if __name__ == "__main__":
    test = ConfusionLoopTest()
    if test.run_test():
        print("ConfusionLoop passed the test successfully!")
    else:
        print("ConfusionLoop failed the test.")
