import shutil
import os
from src.main.resources.CreateLogger import CreateLogger
import pdoc
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import random


create_logger = CreateLogger("PictureReplicator")
logger = create_logger.return_logger()

__pdoc__ = {
    "PictureReplicator": False,   # Exclude this class from documentation
    "__init__": False,  # Exclude this function from documentation
}


class PictureAugmentor:
    def __init__(self):
        # Define the source image file
        self.source_image = "/Users/leon/Uni/Master/Projektarbeit/SDImages/Technology1971_0.png"  # Replace with your image's path
        self.destination_folder = "/Users/leon/Uni/Master/Projektarbeit/SDImages/AugmentedImages"  # Replace with the folder where copies will be saved

    def random_zoom(self, image):
        """
        Apply random zoom by cropping a small portion and resizing back.
        """
        width, height = image.size
        max_zoom = 0.05  # Maximum zoom percentage (5%)
        zoom_factor = random.uniform(0, max_zoom)

        # Calculate cropping box
        crop_width = int(width * zoom_factor)
        crop_height = int(height * zoom_factor)
        left = random.randint(0, crop_width)
        top = random.randint(0, crop_height)
        right = width - random.randint(0, crop_width)
        bottom = height - random.randint(0, crop_height)

        # Crop and resize
        cropped_image = image.crop((left, top, right, bottom))
        zoomed_image = cropped_image.resize((width, height), Image.LANCZOS)
        return zoomed_image

    def change_resolution(self, image):
        """
        Change resolution of the image by resizing to a random size and back.
        """
        width, height = image.size
        resolution_factor = random.uniform(0.8, 1.2)  # Resize by ±20%
        new_width = int(width * resolution_factor)
        new_height = int(height * resolution_factor)

        # Resize to new resolution and back to original
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        final_image = resized_image.resize((width, height), Image.LANCZOS)
        return final_image

    def augment_image(self, image):
        """
        Apply random augmentations to the given image.
        """
        # Random brightness adjustment
        if random.random() < 0.5:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.uniform(0.5, 1.5))  # Brightness factor (0.5 to 1.5)

        # Random color adjustment
        if random.random() < 0.5:
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(random.uniform(0.5, 1.5))  # Color factor (0.5 to 1.5)

        # Random saturation adjustment (simulate saturation by blending)
        if random.random() < 0.5:
            grayscale = ImageOps.grayscale(image).convert("RGB")
            blend_factor = random.uniform(0.3, 0.7)  # Blend factor (higher means less saturation)
            image = Image.blend(image, grayscale, blend_factor)

        # Random RGB channel enhancements
        if random.random() < 0.5:
            channels = image.split()
            channels = [ch.point(lambda i: i * random.uniform(0.8, 1.2)) for ch in channels]
            image = Image.merge("RGB", channels)

        # Random color filter (add hue or tint effects)
        if random.random() < 0.5:
            color_filter = random.choice(["R", "G", "B"])
            filter_factor = random.uniform(1.2, 1.5)
            r, g, b = image.split()
            if color_filter == "R":
                r = r.point(lambda i: i * filter_factor)
            elif color_filter == "G":
                g = g.point(lambda i: i * filter_factor)
            elif color_filter == "B":
                b = b.point(lambda i: i * filter_factor)
            image = Image.merge("RGB", (r, g, b))

        # Random zoom
        if random.random() < 0.5:
            image = self.random_zoom(image)

        # Change resolution
        if random.random() < 0.5:
            image = self.change_resolution(image)

        # Random flip (horizontal or vertical)
        if random.random() < 0.5:
            image = ImageOps.mirror(image)  # Horizontal flip
        if random.random() < 0.5:
            image = ImageOps.flip(image)  # Vertical flip

        # Random rotation
        if random.random() < 0.5:
            angle = random.choice([90, 180, 270])
            image = image.rotate(angle)

        return image

    def __call__(self, num_copies=5):
        """
        Generate augmented image copies.
        """
        # Ensure the destination folder exists
        os.makedirs(self.destination_folder, exist_ok=True)

        # Open the source image
        with Image.open(self.source_image) as img:
            for i in range(num_copies):
                # Apply augmentations
                augmented_image = self.augment_image(img.copy())

                # Save the augmented image
                new_filename = f"Technology1971_{i + 1}.png"  # Create a new filename
                new_filepath = os.path.join(self.destination_folder, new_filename)
                augmented_image.save(new_filepath)
                print(f"Created {new_filepath}")

        print(f"Done creating {num_copies} augmented copies.")


if __name__ == "__main__":
    pic_augmentor = PictureAugmentor()
    pic_augmentor(500)

