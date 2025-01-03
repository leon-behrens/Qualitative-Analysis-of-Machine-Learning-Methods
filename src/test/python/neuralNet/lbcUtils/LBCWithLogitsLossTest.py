import torch
from src.main.python.neuralNet.lbcUtils.LBCWithLogitsLoss import LBCWithLogitsLoss
from src.main.resources.CreateLogger import CreateLogger

# Initialize the logger
create_logger = CreateLogger("LBCWithLogitsLossTest")
logger = create_logger.return_logger()

class LBCWithLogitsLossTest:
    def __init__(self):
        # Test parameters
        self.n_categories_total = 10  # Total number of categories
        self.subset = [0, 2, 4, 6, 8]  # Indices of subset categories
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # Compute device

        # Instantiate the LBCWithLogitsLoss class
        self.loss_fn = LBCWithLogitsLoss(
            n_categories_total=self.n_categories_total,
            subset=self.subset,
            device=self.device
        )

    def run_test(self):
        try:
            # Verify weight computation
            print("Testing Weight Computation...")
            pos_weight = self.loss_fn.pos_weight
            print(f"Calculated Positive Weights: {pos_weight}")

            # Ensure weights have the correct shape
            assert pos_weight.shape == (len(self.subset),), \
                f"Unexpected weights shape: {pos_weight.shape}. Expected: {(len(self.subset),)}"

            # Create mock logits and targets
            batch_size = 8
            logits = torch.randn(batch_size, len(self.subset), device=self.device)
            targets = torch.randint(0, 2, (batch_size, len(self.subset)), dtype=torch.float32, device=self.device)

            # Compute loss
            print("Testing Loss Computation...")
            loss = self.loss_fn(logits, targets)
            print(f"Calculated Loss: {loss.item()}")

            # Ensure the loss is a scalar tensor
            assert loss.dim() == 0, f"Loss should be a scalar. Got: {loss.dim()} dimensions."

            print("All tests passed successfully!")
            return True

        except Exception as e:
            print(f"Error during testing: {e}")
            logger.error(f"Test failed: {e}")
            return False

if __name__ == "__main__":
    test = LBCWithLogitsLossTest()
    if test.run_test():
        print("LBCWithLogitsLoss passed the test successfully!")
    else:
        print("LBCWithLogitsLoss failed the test.")
