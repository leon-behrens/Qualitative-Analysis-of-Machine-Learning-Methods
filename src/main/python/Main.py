from imageCreation.GenerateSDImages import GenerateSDImages
import torch
from src.main.resources.CreateLogger import CreateLogger


create_logger = CreateLogger("Main")
logger = create_logger.return_logger()

__pdoc__ = {
    "main": False,  # Exclude this function from documentation
}
def main():
    logger.info("Main start")
    try:
        # Check if CUDA is available and set the device accordingly
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("device: {}".format(device))

        # Instantiate the GenerateSDImages class
        createImages = GenerateSDImages(
            1,
            "Technology",
            device,
            "/Users/leon/Uni/Master/Projektarbeit/Projektarbeit/SDImages"
        )
        logger.info("Instance of GenerateSDImages created")


        # Generate images for the years 1900 to 1991
        createImages.generate_n_pictures(1900, 1991)
        logger.info("Images generated")

    except Exception as e:
        # Log the exception with an error message
        logger.error(f"An error occurred in main: {e}", exc_info=True)

    logger.info("Main end")

if __name__ == '__main__':
    main()
