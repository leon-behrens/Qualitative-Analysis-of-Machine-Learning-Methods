import shutil
import random
import os

from main.python.neuralNet.CustomDataset import CustomDataset
from main.resources.CreateLogger import CreateLogger



create_logger = CreateLogger("MainDistributePicturesFX1")
logger = create_logger.return_logger()



class MainDistributePicturesFX1:
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
        self.train_folder = os.path.join(destination_folder, "tech_train_FX1")
        self.test_folder = os.path.join(destination_folder, "tech_eval_FX1")

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
            print("FX1: Image added to Train Folder")
            shutil.copy(os.path.join(self.source_folder, image), os.path.join(self.train_folder, image))

        for image in test_images:
            print("FX1: Image added to Eval Folder")
            shutil.copy(os.path.join(self.source_folder, image), os.path.join(self.test_folder, image))

        print(f"Split {len(images)} images: {len(train_images)} to train and {len(test_images)} to test.")

    def __call__(self):
        """
        Execute the sorting process.
        """
        self.create_folders()
        self.split_images()

if __name__ == "__main__":
    distribution = MainDistributePicturesFX1(
        source_folder="/scicore/home/bruder/behleo00/PA/src/main/resources/data/pictures/SD35Images2",  # Replace with the actual source folder path
        destination_folder="/scicore/home/bruder/behleo00/PA/src/main/resources/data/pictures/",  # Replace with the actual destination folder path
        train_ratio=0.8  # 70% images for training, 30% for testing
    )
    distribution()




