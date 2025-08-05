import os
import torch
from diffusers import FluxPipeline
from main.resources.CreateLogger import CreateLogger

create_logger = CreateLogger("GenerateSDImages-Flux1fast")
logger = create_logger.return_logger()

class GenerateSDImagesFlux1fast:
    """
    A class to generate images using the Black Forest Labs FLUX.1 fast model from local files 
    and save them to a specified folder.
    """

    def __init__(self, number_of_images_per_year, prompt, device, path_to_images):
        """
        Initializes the GenerateSDImagesFlux1fast class with the model, device, and saving parameters.

        Args:
            number_of_images_per_year (int): Number of images to generate per year.
            prompt (str): The prompt used to generate images.
            device (str): The device to run the model ('cpu' or 'cuda').
            path_to_images (str): The folder path where generated images will be saved.
        """
        self.device = device
        self.model_path = "/scicore/home/bruder/behleo00/PA/src/main/python/imageCreation/models/flux1-schnell"

        # Ensure the model path exists
        if not os.path.exists(self.model_path):
            logger.error(f"FLUX1: Model path does not exist: {self.model_path}")
            raise FileNotFoundError(f"FLUX1: Model path not found: {self.model_path}")

        try:
            logger.info("FLUX1: Loading FLUX.1 fast model from local directory...")
            self.pipe = FluxPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )

            # Optimize memory usage
            if self.device == "cuda":
                self.pipe.to("cuda")  # Move model to GPU
                self.pipe.enable_model_cpu_offload()  # Offload layers to CPU
                self.pipe.enable_attention_slicing()  # Reduce peak memory usage
                self.pipe.enable_vae_slicing()  # Optimize VAE processing

            logger.info("FLUX1: FluxPipeline (FLUX.1 fast) loaded successfully from local files.")

        except Exception as e:
            logger.error(f"FLUX1: Failed to load the FLUX.1 fast model from local directory: {e}")
            raise RuntimeError("FLUX1: Model initialization failed.") from e

        self.NUMBER_OF_IMAGES_PER_YEAR = number_of_images_per_year
        self.PROMPT = prompt
        self.SAVE_FOLDER = path_to_images
        os.makedirs(self.SAVE_FOLDER, exist_ok=True)  # Ensure the save folder exists
        logger.info("FLUX1: GenerateSDImagesFlux1fast initialized.")

    def generate_n_pictures(self, startYear, endYear):
        """
        Generates and saves images for each year in the specified range.

        For each year in the range [startYear, endYear], this function generates images based on the prompt
        and saves them to the SAVE_FOLDER. The filenames are formatted as '{PROMPT}{YEAR}_{N}.png' 
        to avoid overwriting existing files.

        Args:
            startYear (int): The starting year for image generation.
            endYear (int): The ending year for image generation.

        Raises:
            RuntimeError: If there's an issue with the model or image generation.
        """
        logger.info("FLUX1: generate_n_pictures started.")
        
        for YEAR in range(startYear, endYear + 1):
            for i in range(self.NUMBER_OF_IMAGES_PER_YEAR):
                N = 0
                while os.path.isfile(f"{self.SAVE_FOLDER}/{self.PROMPT}{YEAR}_{N}.png"):
                    N += 1

                if N < self.NUMBER_OF_IMAGES_PER_YEAR:
                    try:
                        # Free GPU memory before generation
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()

                        generator = torch.Generator("cuda").manual_seed(N * 3000 + YEAR)

                        prompt = f"{self.PROMPT} of the year {YEAR}"
                        image = self.pipe(
                            prompt=prompt,
                            guidance_scale=0.0,
                            num_inference_steps=2,  # Reduce steps to lower memory usage
                            max_sequence_length=256,
                            generator=generator
                        ).images[0]

                        image.save(f"{self.SAVE_FOLDER}/{self.PROMPT}{YEAR}_{N}.png")
                        logger.info(f"FLUX1: Saved image: {self.SAVE_FOLDER}/{self.PROMPT}{YEAR}_{N}.png")

                    except Exception as e:
                        logger.error(f"FLUX1: Failed to generate or save image for year {YEAR}, iteration {N}: {e}")

        logger.info("FLUX1: generate_n_pictures finished.")

if __name__ == '__main__':
    createImages = GenerateSDImagesFlux1fast(
        number_of_images_per_year=1,
        prompt="Technology",
        device="cuda" if torch.cuda.is_available() else "cpu",
        path_to_images="./test"
    )
    createImages.generate_n_pictures(1900, 1901)
