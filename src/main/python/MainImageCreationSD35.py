from main.python.imageCreation.GenerateSDImagesSD35 import GenerateSDImagesSD35
import torch
from main.resources.CreateLogger import CreateLogger
import sys

create_logger = CreateLogger("MainImageCreationFLUX1")
logger = create_logger.return_logger()

def main():
    logger.info("Main start")
    torch.cuda.empty_cache()  # Free up unused GPU memory

    try:
        # Get run ID from command-line arguments
        run_id = int(sys.argv[1])

        # Calculate year range based on run ID
        start_year = 1900 + run_id
        end_year = start_year + 1
        logger.info(f"Processing run_id: {run_id}, years: {start_year}-{end_year}")

        print(torch.cuda.is_available())

        # Check if CUDA is available and set the device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Device: {device}")

        create_images_SD35 = GenerateSDImagesSD35(
            number_of_images_per_year=30,
            prompt="Technology",
            device=device,
            path_to_images="/scicore/home/bruder/behleo00/PA/src/main/resources/data/pictures/SD35Images2"
        )
        logger.info("Instance of GenerateSDImagesSD35 created.")


        # Generate images for the specified year range
        create_images_SD35.generate_n_pictures(start_year, end_year)
        logger.info("Images generated")

    except Exception as e:
        logger.error(f"An error occurred in main: {e}", exc_info=True)

    logger.info("Main end")

if __name__ == '__main__':
    main()
