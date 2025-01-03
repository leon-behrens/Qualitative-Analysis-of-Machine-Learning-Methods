import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from src.main.python.neuralNet.CustomDataset import CustomDataset
from src.main.resources.CreateLogger import CreateLogger

# Initialize the logger
create_logger = CreateLogger("CustomDatasetTest")
logger = create_logger.return_logger()

class CustomDatasetTest:
    def __init__(self):
        # Create a mock directory and sample files for testing
        self.mock_dir = "mock_data"
        self.setup_mock_data()

        # Define a simple transform
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

        # Instantiate the CustomDataset
        self.dataset = CustomDataset(directory=self.mock_dir, transform=self.transform)
        self.dataloader = DataLoader(self.dataset, batch_size=2, shuffle=True, drop_last=True)


    def setup_mock_data(self):
        """Sets up mock data for testing."""
        if not os.path.exists(self.mock_dir):
            os.makedirs(self.mock_dir)

        # Create sample image files
        sample_filenames = [
            "Technology2000_1.jpg",
            "Technology1995_2.jpg",
            "Technology2010_3.jpg"
        ]
        for filename in sample_filenames:
            filepath = os.path.join(self.mock_dir, filename)
            if not os.path.exists(filepath):
                img = Image.new('RGB', (100, 100), color=(255, 0, 0))
                img.save(filepath)

    def cleanup_mock_data(self):
        """Cleans up the mock data directory."""
        if os.path.exists(self.mock_dir):
            for f in os.listdir(self.mock_dir):
                os.remove(os.path.join(self.mock_dir, f))
            os.rmdir(self.mock_dir)

    def run_test(self):
        """Runs all tests for the CustomDataset class."""
        try:
            print("Testing CustomDataset...")

            # Test length
            assert len(self.dataset) == 3, f"Expected dataset length 3, got {len(self.dataset)}."

            # Test filename_to_year
            filename = "Technology1995_sample.jpg"
            expected_year = 1995 - 1900
            extracted_year = self.dataset.filename_to_year(filename, self.dataset.pattern)
            assert extracted_year == expected_year, \
                f"Expected year {expected_year}, got {extracted_year}."

            # Test __getitem__
            for i in range(len(self.dataset)):
                img, target = self.dataset[i]
                assert isinstance(img, torch.Tensor), "Image is not a tensor."
                assert img.shape[0] == 3, "Image does not have 3 channels."
                assert isinstance(target, torch.Tensor), "Target is not a tensor."
                assert target.dtype == torch.int64, "Target is not of type int64."

            # Test DataLoader
            for batch_imgs, batch_targets in self.dataloader:
                # Allow for partial batches
                assert batch_imgs.shape[0] <= 2, f"Unexpected batch size: {batch_imgs.shape[0]}."
                assert batch_imgs.shape[1] == 3, "Image channels mismatch."
                assert batch_targets.dtype == torch.int64, "Batch targets dtype mismatch."
                print(f"Batch Images Shape: {batch_imgs.shape}, Batch Targets: {batch_targets}")

            print("All tests passed successfully!")
            return True

        except Exception as e:
            print(f"Error during testing: {e}")
            logger.error(f"Test failed: {e}")
            return False
        finally:
            self.cleanup_mock_data()

if __name__ == "__main__":
    test = CustomDatasetTest()
    if test.run_test():
        print("CustomDataset passed the test successfully!")
    else:
        print("CustomDataset failed the test.")
