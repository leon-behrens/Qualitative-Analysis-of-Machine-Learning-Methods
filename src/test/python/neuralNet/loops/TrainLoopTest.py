import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import SGD
from src.main.python.neuralNet.lbcUtils.LBCLabel import LBCLabel
from src.main.python.neuralNet.lbcUtils.LBCWithLogitsLoss import LBCWithLogitsLoss
from src.main.python.neuralNet.loops.TrainLoop import TrainLoop
from src.main.resources.CreateLogger import CreateLogger
from src.main.python.neuralNet.SimpleCNN import SimpleCNN

# Initialize the logger
create_logger = CreateLogger("TrainLoopTest")
logger = create_logger.return_logger()

class TrainLoopTest:
    def __init__(self):
        # Test parameters
        self.n_categories = 10  # Number of categories
        self.subset = list(range(self.n_categories-1))  # Subset of categories
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # Compute device
        self.record_every = 10  # Interval for recording loss

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

        # Optimizer
        self.optimizer = SGD(self.model.parameters(), lr=0.01)

        # Instantiate TrainLoop
        self.train_loop = TrainLoop(
            dataloader=self.dataloader,
            model=self.model,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            device=self.device,
            subset=self.subset,
            record_every=self.record_every
        )

    def run_test(self):
        try:
            # Run the training loop
            print("Running TrainLoop...")
            losses = self.train_loop()

            # Validate losses
            print(f"Recorded Losses: {losses}")
            assert len(losses) > 0, "No losses recorded. The training loop may not have run correctly."
            assert all(isinstance(loss, float) for loss in losses), "Losses should be a list of floats."

            print("TrainLoop ran successfully!")
            return True

        except Exception as e:
            print(f"Error during testing: {e}")
            logger.error(f"Test failed: {e}")
            return False

if __name__ == "__main__":
    test = TrainLoopTest()
    if test.run_test():
        print("TrainLoop passed the test successfully!")
    else:
        print("TrainLoop failed the test.")
