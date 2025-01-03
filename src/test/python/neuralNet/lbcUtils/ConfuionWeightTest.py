import torch
from src.main.python.neuralNet.lbcUtils.ConfusionWeight import ConfusionWeight
from src.main.resources.CreateLogger import CreateLogger

# Initialize the logger
create_logger = CreateLogger("ConfusionWeightTest")
logger = create_logger.return_logger()


class ConfusionWeightTest:
    def __init__(self):
        # Test parameters
        self.n_categories_total = 10  # Total number of categories
        self.subset = [0, 2, 4, 6, 8]  # Indices of subset categories
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # Compute device

        # Instantiate the ConfusionWeight class
        self.confusion_weight = ConfusionWeight(
            n_categories_total=self.n_categories_total,
            subset=self.subset,
            device=self.device
        )

    def run_test(self):
        # Perform the weight calculation
        try:
            weights = self.confusion_weight()
            print("ConfusionWeight initialized and calculation successful!")
            print(f"Calculated weights: {weights}")

            # Verify the weights tensor
            assert weights.shape == (len(self.subset),), \
                f"Unexpected weights shape: {weights.shape}. Expected: {(len(self.subset),)}"
            assert weights.device.type == self.device, \
                f"Unexpected device: {weights.device.type}. Expected: {self.device}"

            # Verify the weight values are correctly normalized
            expected_weights = torch.arange(1, self.n_categories_total + 1, device=self.device)[
                                   self.subset] / self.n_categories_total
            assert torch.allclose(weights, expected_weights), \
                f"Weights do not match expected values. Got: {weights}, Expected: {expected_weights}"

            return True
        except Exception as e:
            print(f"Error during testing: {e}")
            logger.error(f"Test failed: {e}")
            return False


if __name__ == "__main__":
    test = ConfusionWeightTest()
    if test.run_test():
        print("ConfusionWeight passed the test successfully!")
    else:
        print("ConfusionWeight failed the test.")
