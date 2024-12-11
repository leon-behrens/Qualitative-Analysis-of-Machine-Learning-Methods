import os
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from src.main.resources.CreateLogger import CreateLogger

create_logger = CreateLogger("GenerateSDImages")
logger = create_logger.return_logger()

class GenerateSDImages:
    """
    A class to generate images using the Stable Diffusion model and save them to a specified folder.

    Attributes:
        device (str): The device to run the model on ('cpu' or 'cuda').
        model_id (str): The model identifier for the Stable Diffusion model.
        pipe (StableDiffusionPipeline): The pipeline for generating images.
        NUMBER_OF_IMAGES_PER_YEAR (int): The number of images to generate per year.
        PROMPT (str): The prompt for generating the images.
        SAVE_FOLDER (str): The path to the folder where images will be saved.
    """

    def __init__(self, number_of_images_per_year, prompt, device, path_to_images):
        """
        Initializes the GenerateSDImages class with the model, device, and saving parameters.

        Args:
            number_of_images_per_year (int): The number of images to generate for each year.
            prompt (str): The prompt used to generate the images.
            device (str): The device to run the model ('cpu' or 'cuda').
            path_to_images (str): The folder path where generated images will be saved.
        """
        self.device: str = device
        self.model_id: str = "stabilityai/stable-diffusion-2-1"

        try:
            self.pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(self.model_id, torch_dtype=torch.float32)
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
            self.pipe = self.pipe.to(self.device)
            logger.info("StableDiffusionPipeline loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load the Stable Diffusion model: {e}")
            raise RuntimeError("Model initialization failed.") from e

        self.NUMBER_OF_IMAGES_PER_YEAR: int = number_of_images_per_year
        self.PROMPT: str = prompt
        self.SAVE_FOLDER: str = path_to_images
        logger.info("GenerateSDImages initialized.")



    def generate_n_pictures(self, startYear, endYear):
        """
        Generates and saves images for each year in the specified range.

        For each year in the range [startYear, endYear], this function generates images based on the prompt
        and saves them to the SAVE_FOLDER. The filenames are formatted as '{PROMPT}{YEAR}_{N}.png' to avoid
        overwriting existing files.

        Args:
            startYear (int): The starting year for image generation.
            endYear (int): The ending year for image generation.

        Raises:
            RuntimeError: If there's an issue with the model or image generation.
        """
        logger.info("generate_n_pictures started.")
        # Create the save folder if it doesn't exist
        os.makedirs(self.SAVE_FOLDER, exist_ok=True)

        # Loop to generate images for each year
        for YEAR in range(startYear, endYear + 1):
            for i in range(self.NUMBER_OF_IMAGES_PER_YEAR):
                N = 0
                # Check if a file with the current name already exists and increment N to avoid overwriting
                while os.path.isfile(f"{self.SAVE_FOLDER}/{self.PROMPT}{YEAR}_{N}.png"):
                    N += 1

                # Generate the image if the limit hasn't been reached
                if N < self.NUMBER_OF_IMAGES_PER_YEAR:
                    try:
                        generator = torch.Generator(self.device).manual_seed(N * 3000 + YEAR)
                        prompt = f"{self.PROMPT} of the year {YEAR}"
                        image = self.pipe(prompt, generator=generator).images[0]
                        image.save(f"{self.SAVE_FOLDER}/{self.PROMPT}{YEAR}_{N}.png")
                        print(f"Saved image: {self.SAVE_FOLDER}/{self.PROMPT}{YEAR}_{N}.png")
                        logger.info(f"Saved image: {self.SAVE_FOLDER}/{self.PROMPT}{YEAR}_{N}.png")
                    except Exception as e:
                        logger.error(f"Failed to generate or save image for year {YEAR}, iteration {N}: {e}")

        logger.info("generate_n_pictures finished.")

