import os
from PIL import Image, UnidentifiedImageError
import re
import torch
from src.main.resources.CreateLogger import CreateLogger

create_logger = CreateLogger("CustomDataset")
logger = create_logger.return_logger()

class CustomDataset:
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
        """
        logger.info("Initializing CustomDataset starts")
        self.directory = directory
        self.transform = transform
        try:
            self.file_list = [f for f in os.listdir(self.directory)]
        except FileNotFoundError as e:
            logger.error(f"Directory not found: {self.directory}")
            raise FileNotFoundError(f"Directory not found: {self.directory}") from e
        self.pattern = r'technology(-?\d+)_'
        logger.info("Initializing CustomDataset ends")

    def __len__(self):
        """
        Returns the number of files in the dataset.

        Returns:
            int: The number of image files in the dataset.
        """
        logger.info("__len__")
        return len(self.file_list)

    def filename_to_year(self, filename):
        """
        Extracts the year from the filename using a regex pattern.

        Args:
            filename (str): The filename to extract the year from.

        Returns:
            int: The extracted year adjusted by adding -1900.

        Raises:
            ValueError: If the year cannot be extracted from the filename.
        """
        logger.info("__filename_to_year starts")
        match = re.search(self.pattern, filename)
        if match is None:
            logger.error(f"filename_to_year: could not get year from {filename}")
            raise ValueError(f"Could not get year from {filename}")
        logger.info("__filename_to_year ends")
        return int(match.group(1)) + -1900

    def __getitem__(self, index):
        """
        Retrieves the image and target label at the specified index.

        Args:
            index (int): The index of the file to retrieve.

        Returns:
            tuple: A tuple containing the image and the corresponding target label.

        Raises:
            FileNotFoundError: If the image file does not exist.
            ValueError: If there is an issue extracting the year from the filename.
            RuntimeError: If the image cannot be loaded or converted.
        """
        logger.info("__getitem__ starts")
        filename = self.file_list[index]

        # Load image
        img_path = os.path.join(self.directory, filename)
        try:
            with open(img_path, 'rb') as f:
                img = Image.open(f).convert('RGB')
        except FileNotFoundError as e:
            logger.error(f"Image file not found: {img_path}")
            raise FileNotFoundError(f"Image file not found: {img_path}") from e
        except UnidentifiedImageError as e:
            logger.error(f"Cannot identify image file: {img_path}")
            raise RuntimeError(f"Cannot identify image file: {img_path}") from e
        except Exception as e:
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
            year = self.filename_to_year(filename)
        except ValueError as e:
            logger.error(f"Error extracting year from filename: {filename}")
            raise ValueError(f"Error extracting year from filename: {filename}") from e

        target = torch.tensor(year, dtype=torch.int64)

        logger.info("__getitem__ ends")
        return img, target