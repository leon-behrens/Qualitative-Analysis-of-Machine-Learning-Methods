import torch
from torchsummary import summary
from src.main.python.neuralNet.SimpleCNN import SimpleCNN
from src.main.resources.CreateLogger import CreateLogger

# Initialize the logger
create_logger = CreateLogger("SimpleCNNTest")
logger = create_logger.return_logger()

class SimpleCNNTest:
    def __init__(self):
        # Test parameters
        self.img_dim = 60  # Dimension of the input image (assumed square)
        self.n_categories = 200  # Number of output categories/classes
        self.batch_size = 8  # Number of images in a batch

        # Instantiate the model
        self.model = SimpleCNN(img_dim=self.img_dim, n_categories=self.n_categories)

    def run_test(self):
        # Display model summary
        print("Model Summary:")
        summary(self.model, (1, self.img_dim, self.img_dim))

        # Check GPU compatibility
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)

        # Create mock input data
        mock_data = torch.randn(self.batch_size, 1, self.img_dim, self.img_dim).to(device)

        # Perform a forward pass on the specified device
        try:
            output = self.model(mock_data)
            print("Model initialized and forward pass successful!")
            print(f"Input shape: {mock_data.shape}")
            print(f"Output shape: {output.shape}")
            print(f"Tested on device: {device}")

            # Verify output shape
            assert output.shape == (self.batch_size, self.n_categories - 1), \
                f"Unexpected output shape: {output.shape}. Expected: {(self.batch_size, self.n_categories - 1)}"
            return True
        except Exception as e:
            print(f"Error during testing: {e}")
            logger.error(f"Test failed: {e}")
            return False

if __name__ == "__main__":
    test = SimpleCNNTest()
    if test.run_test():
        print("SimpleCNN passed the test successfully!")
    else:
        print("SimpleCNN failed the test.")
