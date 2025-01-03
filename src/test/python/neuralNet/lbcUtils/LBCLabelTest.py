import torch
from src.main.python.neuralNet.lbcUtils.LBCLabel import LBCLabel
from src.main.resources.CreateLogger import CreateLogger

# Initialize the logger
create_logger = CreateLogger("LBCLabelTest")
logger = create_logger.return_logger()

class LBCLabelTest:
    def __init__(self):
        # Test parameters
        self.y = torch.tensor([1, 3, 5], dtype=torch.int64)  # Sample labels
        self.subset = [0, 1, 2, 3, 4, 5]  # Subset of categories
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # Compute device

        # Instantiate the LBCLabel class
        self.lbclabel = LBCLabel(
            subset=self.subset,
            device=self.device
        )

    def run_test(self):
        try:
            # Perform label vector conversion
            print("Testing Label Vector Conversion...")
            result = self.lbclabel(self.y.to(self.device))
            print(f"Input Labels (y): {self.y}")
            print(f"Subset: {self.subset}")
            print(f"Converted Label Vectors:\n{result}")

            # Verify the shape of the result
            assert result.shape == (len(self.y), len(self.subset)), \
                f"Unexpected result shape: {result.shape}. Expected: ({len(self.y)}, {len(self.subset)})"

            # Verify the boolean tensor correctness
            expected_result = torch.tensor(
                [[True, False, False, False, False, False],
                 [True, True, True, False, False, False],
                 [True, True, True, True, True, False]],
                device=self.device
            )
            assert torch.equal(result, expected_result), \
                f"Result does not match expected values.\nGot:\n{result}\nExpected:\n{expected_result}"

            print("All tests passed successfully!")
            return True

        except Exception as e:
            print(f"Error during testing: {e}")
            logger.error(f"Test failed: {e}")
            return False

if __name__ == "__main__":
    test = LBCLabelTest()
    if test.run_test():
        print("LBCLabel passed the test successfully!")
    else:
        print("LBCLabel failed the test.")
