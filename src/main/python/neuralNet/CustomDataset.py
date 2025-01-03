import os
from PIL import Image, UnidentifiedImageError
import re
import torch
from src.main.resources.CreateLogger import CreateLogger
from torch.utils.data import Dataset

create_logger = CreateLogger("CustomDataset")
logger = create_logger.return_logger()

class CustomDataset(Dataset):
    """
    A custom dataset class for loading images and their corresponding labels based on filenames.

    Attributes:
        directory (str): Path to the directory containing image files.
        transform (callable, optional): Optional transform to be applied to the images.
        file_list (list): List of filenames in the directory.
        pattern (str): Regular expression pattern to extract year information from filenames.
    """

    def __init__(self, directory, transform=None):
        """
        Initializes the CustomDataset.

        Args:
            directory (str): Path to the directory containing the image files.
            transform (callable, optional): A function/transform to apply to the images.
            extensions (tuple): Allowed image file extensions.
        """
        logger.info("Initializing CustomDataset starts")
        self.directory = directory
        self.transform = transform
        self.extensions = ('.png', '.jpg', '.jpeg', '.bmp')

        try:
            self.file_list = [
                f for f in os.listdir(self.directory)
                if f.lower().endswith(self.extensions)
            ]
            if not self.file_list:
                logger.error("No valid image files found in the specified directory.")
                raise FileNotFoundError("No valid image files found in the specified directory.")
        except FileNotFoundError as e:
            logger.error(f"Directory not found: {self.directory}")
            raise FileNotFoundError(f"Directory not found: {self.directory}") from e

        self.pattern = r'[Tt]echnology(-?\d+)_'
        logger.info("Initializing CustomDataset ends")

    def __len__(self):
        """Returns the number of files in the dataset."""
        logger.info("__len__")
        try:
            return len(self.file_list)
        except TypeError as e:
            logger.error(e)
            raise e

    def filename_to_year(self, filename, pattern):
        """
        Extracts the year from the filename using a regex pattern.

        Args:
            filename (str): The filename to extract the year from.

        Returns:
            int: The extracted year adjusted by adding -1900.

        Raises:
            ValueError: If the year cannot be extracted from the filename.
        """
        match = re.search(pattern, filename)
        if match is None:
            logger.error(f"filename_to_year: could not get year from {filename}")
            raise ValueError(f"Could not get year from {filename}")
        return int(match.group(1)) + -1900




    def __getitem__(self, index):
        """
        Retrieves the image and target label at the specified index.

        Args:
            index (int): The index of the file to retrieve.

        Returns:
            tuple: A tuple containing the image and the corresponding target label.
        """
        filename = self.file_list[index]

        # Load image
        img_path = os.path.join(self.directory, filename)
        try:
            with open(img_path, 'rb') as f:
                img = Image.open(f).convert('RGB')
        except (FileNotFoundError, UnidentifiedImageError) as e:
            logger.error(f"Error loading image: {img_path}")
            raise RuntimeError(f"Error loading image: {img_path}") from e

        # Apply transform if provided
        if self.transform:
            try:
                img = self.transform(img)
            except Exception as e:
                logger.error(f"Error applying transform to image: {img_path}")
                raise RuntimeError(f"Error applying transform to image: {img_path}") from e

        # Load label and convert to tensor
        try:
            year = self.filename_to_year(filename, self.pattern)
        except ValueError as e:
            logger.error(f"Error extracting year from filename: {filename}")
            raise ValueError(f"Error extracting year from filename: {filename}") from e

        target = torch.tensor(year, dtype=torch.int64)

        return img, target
