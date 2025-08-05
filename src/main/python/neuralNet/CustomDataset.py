import os
from PIL import Image, UnidentifiedImageError
import re
import torch
from torch.utils.data import Dataset
from main.resources.CreateLogger import CreateLogger

# Create a logger instance
create_logger = CreateLogger("CustomDataset")
logger = create_logger.return_logger()

class CustomDataset(Dataset):
    """
    A custom dataset class for loading images and their corresponding labels based on filenames.
    """

    def __init__(self, directory, transform=None, extensions=None):
        """
        Initializes the CustomDataset.

        Args:
            directory (str): Path to the directory containing the image files.
            transform (callable, optional): A function/transform to apply to the images.
            extensions (tuple, optional): Allowed image file extensions. Defaults to common image types.
        """
        logger.info("Initializing CustomDataset starts")
        self.directory = directory
        self.transform = transform
        self.extensions = extensions or ('.png', '.jpg', '.jpeg', '.bmp')

        # Validate the directory
        if not os.path.isdir(self.directory):
            logger.error(f"Invalid directory: {self.directory}")
            raise FileNotFoundError(f"Invalid directory: {self.directory}")

        # List image files
        self.file_list = [
            f for f in os.listdir(self.directory)
            if f.lower().endswith(self.extensions)
        ]
        if not self.file_list:
            logger.error("No valid image files found in the specified directory.")
            raise FileNotFoundError("No valid image files found in the specified directory.")

        # Set regex pattern for year extraction
        self.pattern = r'[Tt]echnology(-?\d+)_'
        logger.info(f"CustomDataset initialized with {len(self.file_list)} files.")

    def __len__(self):
        """Returns the number of files in the dataset."""
        return len(self.file_list)

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
            logger.error(f"filename_to_year: could not extract year from {filename}")
            raise ValueError(f"Could not extract year from {filename}")
        try:
            year = int(match.group(1)) + -1900
            logger.debug(f"Extracted year {year} from filename {filename}")
            return year
        except ValueError as e:
            logger.error(f"Invalid year format in filename {filename}")
            raise ValueError(f"Invalid year format in filename {filename}") from e

    def __getitem__(self, index):
        """
        Retrieves the image and target label at the specified index.

        Args:
            index (int): The index of the file to retrieve.

        Returns:
            tuple: A tuple containing the image and the corresponding target label.
        """
        filename = self.file_list[index]
        img_path = os.path.join(self.directory, filename)

        # Load image
        try:
            with open(img_path, 'rb') as f:
                img = Image.open(f).convert('RGB')
        except (FileNotFoundError, UnidentifiedImageError) as e:
            logger.error(f"Error loading image: {img_path}. Error: {e}")
            raise RuntimeError(f"Error loading image: {img_path}") from e

        # Apply transform if provided
        if self.transform:
            try:
                img = self.transform(img)
            except Exception as e:
                logger.error(f"Error applying transform to image {img_path}: {e}")
                raise RuntimeError(f"Error applying transform to image {img_path}") from e

        # Extract year from filename
        try:
            year = self.filename_to_year(filename, self.pattern)
        except ValueError as e:
            logger.error(f"Error extracting year from filename: {filename}")
            raise ValueError(f"Error extracting year from filename: {filename}") from e

        # Convert year to tensor
        target = torch.tensor(year, dtype=torch.int64)

        return img, target
