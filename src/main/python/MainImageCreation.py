from main.python.imageCreation.GenerateSDImages import GenerateSDImages
import torch
from main.resources.CreateLogger import CreateLogger
import sys

create_logger = CreateLogger("Main")
logger = create_logger.return_logger()

def main():
    logger.info("Main start")
    try:
        # Get run ID from command-line arguments
        run_id = int(sys.argv[1])

        # Calculate year range based on run ID
        start_year = 1900 + run_id
        end_year = start_year + 1
        logger.info(f"Processing run_id: {run_id}, years: {start_year}-{end_year}")

        # Check if CUDA is available and set the device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("device: {}".format(device))

        # Instantiate the GenerateSDImages class
        create_images = GenerateSDImages(
            3 * 27,
            "Technology",
            device,
            "/scicore/home/bruder/behleo00/PA/src/main/resources/data/pictures/SDImages"
        )
        logger.info("Instance of GenerateSDImages created")

        # Generate images for the year range
        create_images.generate_n_pictures(start_year, end_year)
        logger.info("Images generated")

    except Exception as e:
        # Log the exception with an error message
        logger.error(f"An error occurred in main: {e}", exc_info=True)

    logger.info("Main end")

if __name__ == '__main__':
    main()
