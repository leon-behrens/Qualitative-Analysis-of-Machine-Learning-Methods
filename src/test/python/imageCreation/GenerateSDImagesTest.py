import unittest
from unittest.mock import patch, MagicMock
from src.main.python.imageCreation.GenerateSDImages import GenerateSDImages
import os


class TestGenerateSDImages(unittest.TestCase):
    """
    Unit test class for GenerateSDImages.
    """

    def setUp(self):
        """
        Set up test parameters before each test case.
        """
        self.number_of_images_per_year = 2
        self.prompt = "Technology"
        self.device = "cpu"
        self.path_to_images = "test_images"
        self.start_year = 2020
        self.end_year = 2021

    @patch("torch.Generator")
    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_initialization(self, mock_from_pretrained, mock_generator):
        """
        Test initialization of GenerateSDImages class.
        """
        # Mock the pipeline
        mock_pipeline = MagicMock()
        mock_from_pretrained.return_value = mock_pipeline

        # Initialize the class
        generator = GenerateSDImages(self.number_of_images_per_year, self.prompt, self.device, self.path_to_images)

        # Assertions
        self.assertEqual(generator.NUMBER_OF_IMAGES_PER_YEAR, self.number_of_images_per_year)
        self.assertEqual(generator.PROMPT, self.prompt)
        self.assertEqual(generator.device, self.device)
        self.assertEqual(generator.SAVE_FOLDER, self.path_to_images)
        mock_from_pretrained.assert_called_once()

    @patch("os.makedirs")
    @patch("os.path.isfile", side_effect=lambda x: False)
    @patch("PIL.Image.Image.save")
    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_generate_n_pictures(self, mock_from_pretrained, mock_save, mock_isfile, mock_makedirs):
        """
        Test generate_n_pictures method.
        """
        # Mock the pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.return_value.images = [MagicMock()]
        mock_from_pretrained.return_value = mock_pipeline

        # Initialize the class
        generator = GenerateSDImages(self.number_of_images_per_year, self.prompt, self.device, self.path_to_images)

        # Call the method
        generator.generate_n_pictures(self.start_year, self.end_year)

        # Assertions
        total_images = (self.end_year - self.start_year + 1) * self.number_of_images_per_year
        self.assertEqual(mock_save.call_count, total_images)
        mock_makedirs.assert_called_once_with(self.path_to_images, exist_ok=True)

    @patch("diffusers.StableDiffusionPipeline.from_pretrained", side_effect=Exception("Model load error"))
    def test_initialization_failure(self, mock_from_pretrained):
        """
        Test that initialization fails properly when model loading raises an exception.
        """
        with self.assertRaises(RuntimeError) as context:
            GenerateSDImages(self.number_of_images_per_year, self.prompt, self.device, self.path_to_images)
        self.assertIn("Model initialization failed", str(context.exception))


if __name__ == "__main__":
    unittest.main()
