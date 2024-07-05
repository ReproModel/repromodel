from PIL import Image, ImageDraw
import numpy as np
import os
import random

######################################################################
# CONSTANTS
######################################################################

# Output directory for generated RGB image data.
rgb_output_dir = 'repromodel_core/data/dummyData/input'

# Output directory for generated binary image data.
binary_output_dir = 'repromodel_core/data/dummyData/target'

# Define size of the images (width, height).
image_size = (128, 128)

# Define number of images.
image_count = 100

# Define maximum number of circles per image.
max_circles = 10

######################################################################
# INITIALIZATION
######################################################################

# Create output directory for generated RGB image data if it doesn't exist.
os.makedirs(rgb_output_dir, exist_ok=True)

# Create output directory for generated binary image data if it doesn't exist.
os.makedirs(binary_output_dir, exist_ok=True)

######################################################################
# DATA GENERATION
######################################################################

for i in range(image_count):
    # Create a random RGB background.
    background_data = np.random.randint(0, 256, (image_size[1], image_size[0], 3), dtype=np.uint8)
    rgb_image = Image.fromarray(background_data, 'RGB')
    draw_rgb = ImageDraw.Draw(rgb_image)

    # Create a blank binary image with a black background.
    binary_image = Image.new('L', image_size, 0)
    draw_binary = ImageDraw.Draw(binary_image)

    # Randomly decide the number of circles to draw.
    num_circles = random.randint(1, max_circles)

    for _ in range(num_circles):
        # Randomly decide circle parameters.
        radius = random.randint(5, 25)
        x = random.randint(0, image_size[0] - 1)
        y = random.randint(0, image_size[1] - 1)
        color = tuple(np.random.randint(0, 256, size=3))

        # Draw the circle on the RGB image.
        draw_rgb.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

        # Draw the circle on the binary image (white circle on black background).
        draw_binary.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)

    # Save images.
    rgb_image.save(os.path.join(rgb_output_dir, f'{i}.png'))
    binary_image.save(os.path.join(binary_output_dir, f'{i}.png'))