import shutil
import random
import os

from src.main.python.neuralNet.CustomDataset import CustomDataset
from src.main.resources.CreateLogger import CreateLogger
import pdoc


create_logger = CreateLogger("MainDistributePictures")
logger = create_logger.return_logger()

__pdoc__ = {
    "MainAnalysis": False,   # Exclude this class from documentation
    "__init__": False,  # Exclude this function from documentation
}

class MainDistributePictures:
    def __init__(self, source_folder, destination_folder, train_ratio=0.8):
        """
        Initialize the PictureSorter with source and destination paths and the train-test split ratio.

        :param source_folder: Path to the folder containing the images.
        :param destination_folder: Path to the folder where the split datasets will be saved.
        :param train_ratio: Proportion of images to include in the training set (default is 0.8).
        """
        self.source_folder = source_folder
        self.destination_folder = destination_folder
        self.train_ratio = train_ratio
        self.train_folder = os.path.join(destination_folder, "tech_train")
        self.test_folder = os.path.join(destination_folder, "tech_eval")

    def create_folders(self):
        """
        Create train and test folders if they don't exist.
        """
        os.makedirs(self.train_folder, exist_ok=True)
        os.makedirs(self.test_folder, exist_ok=True)

    def split_images(self):
        """
        Split images from the source folder into train and test folders based on the train ratio.
        """
        # List all image files in the source folder
        images = [f for f in os.listdir(self.source_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)  # Shuffle the images to ensure randomness

        # Determine split point
        split_point = int(len(images) * self.train_ratio)
        train_images = images[:split_point]
        test_images = images[split_point:]

        # Move images to respective folders
        for image in train_images:
            shutil.copy(os.path.join(self.source_folder, image), os.path.join(self.train_folder, image))

        for image in test_images:
            shutil.copy(os.path.join(self.source_folder, image), os.path.join(self.test_folder, image))

        print(f"Split {len(images)} images: {len(train_images)} to train and {len(test_images)} to test.")

    def __call__(self):
        """
        Execute the sorting process.
        """
        self.create_folders()
        self.split_images()

if __name__ == "__main__":
    distribution = MainDistributePictures()
    distribution()




