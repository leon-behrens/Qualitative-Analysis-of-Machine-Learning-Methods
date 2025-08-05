import os
import torch
from diffusers import StableDiffusion3Pipeline
from main.resources.CreateLogger import CreateLogger

create_logger = CreateLogger("GenerateSDImagesSD35")
logger = create_logger.return_logger()

class GenerateSDImagesSD35:
    """
    A class to generate images using the Stable Diffusion 3.5 Medium model from local files and save them to a specified folder.
    """

    def __init__(self, number_of_images_per_year, prompt, device, path_to_images):
        """
        Initializes the GenerateSDImagesSD35 class with the model, device, and saving parameters.

        Args:
            number_of_images_per_year (int): The number of images to generate for each year.
            prompt (str): The prompt used to generate the images.
            device (str): The device to run the model ('cpu' or 'cuda').
            path_to_images (str): The folder path where generated images will be saved.
        """
        self.device: str = device
        self.model_path: str = os.path.expanduser("~/PA/src/main/python/imageCreation/models/sd3.5")  # Use local path

        # Ensure the model path exists and contains necessary files
        if not os.path.exists(self.model_path):
            logger.error(f"SD3.5: Model path does not exist: {self.model_path}")
            raise FileNotFoundError(f"SD3.5: Model path not found: {self.model_path}")

        try:
            logger.info("SD3.5: Loading Stable Diffusion 3.5 Medium model from local directory...")
            self.pipe: StableDiffusion3Pipeline = StableDiffusion3Pipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32
            )
            self.pipe = self.pipe.to(self.device)
            logger.info("SD3.5: StableDiffusion3Pipeline (SD 3.5 Medium) loaded successfully from local files.")
        except Exception as e:
            logger.error(f"SD3.5: Failed to load the SD 3.5 Medium model from local directory: {e}")
            raise RuntimeError("SD3.5: Model initialization failed.") from e

        self.NUMBER_OF_IMAGES_PER_YEAR: int = number_of_images_per_year
        self.PROMPT: str = prompt
        self.SAVE_FOLDER: str = path_to_images
        logger.info("SD3.5: GenerateSDImagesSD35 initialized.")

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
        logger.info("SD3.5: generate_n_pictures started.")
        os.makedirs(self.SAVE_FOLDER, exist_ok=True)

        for YEAR in range(startYear, endYear + 1):
            for i in range(self.NUMBER_OF_IMAGES_PER_YEAR):
                N = 0
                while os.path.isfile(f"{self.SAVE_FOLDER}/{self.PROMPT}{YEAR}_{N}.png"):
                    N += 1

                if N < self.NUMBER_OF_IMAGES_PER_YEAR:
                    try:
                        generator = torch.manual_seed(N * 3000 + YEAR)
                        prompt = f"{self.PROMPT} of the year {YEAR}"
                        image = self.pipe(
                            prompt=prompt,
                            num_inference_steps=40,
                            guidance_scale=4.5
                        ).images[0]

                        image.save(f"{self.SAVE_FOLDER}/{self.PROMPT}{YEAR}_{N}.png")
                        logger.info(f"FLUX1: Saved image: {self.SAVE_FOLDER}/{self.PROMPT}{YEAR}_{N}.png")
                    except Exception as e:
                        logger.error(f"SD3.5: Failed to generate or save image for year {YEAR}, iteration {N}: {e}")

        logger.info("SD3.5: generate_n_pictures finished.")

if __name__ == '__main__':
    creatImages = GenerateSDImagesSD35(1, "Technology", "cuda" if torch.cuda.is_available() else "cpu", "./test")
