import torch
from torchsummary import summary
from src.main.python.neuralNet.MyModel import MyModel
from src.main.resources.CreateLogger import CreateLogger

# Initialize the logger
create_logger = CreateLogger("MyModelTest")
logger = create_logger.return_logger()

class MyModelTest:
    def __init__(self):
        # Test parameters
        self.n_categories = 130  # Number of categories/grid points
        self.batch_size = 8  # Number of images in a batch
        self.img_dim = 224  # ResNet-50 requires 224x224 input images

        # Instantiate the model
        self.model = MyModel(n_categories=self.n_categories)()

    def run_test(self):
        # Display model summary
        print("Model Summary:")
        summary(self.model, (3, self.img_dim, self.img_dim))

        # Check GPU compatibility
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)

        # Create mock input data
        mock_data = torch.randn(self.batch_size, 3, self.img_dim, self.img_dim).to(device)

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
    test = MyModelTest()
    if test.run_test():
        print("MyModel passed the test successfully!")
    else:
        print("MyModel failed the test.")
