from PIL import Image

import numpy as np
import os



######################################################################
# CONSTANTS
######################################################################


# Output directory for generated RGB image data.
rgb_output_dir = 'repromodel_core/data/dummyData/input'

# Output directory for generated binary image data.
binary_output_dir = 'repromodel_core/data/dummyData/target'

# Define size of the images (width, height).
image_size = (256, 256)

# Define number of images.
image_count = 100



######################################################################
# INITIALIZATION
######################################################################


# Create output directory for generated RGB image data if it doesn't exist.
os.makedirs(rgb_output_dir, exist_ok=True)

# Create output directory for generated binary image data if it doesn't exist.
os.makedirs(binary_output_dir, exist_ok=True)



######################################################################
# DATA - RGB Images
######################################################################


# Generate and save random RGB images.
for i in range(image_count):
    
    # Create a random array of shape (height, width, 3) and dtype uint8.
    data = np.random.randint(0, 256, (image_size[1], image_size[0], 3), dtype=np.uint8)
    
    # Create an image from the numpy array.
    image = Image.fromarray(data, 'RGB')
    
    # Save image.
    image.save(os.path.join(rgb_output_dir, f'{i}.png'))


######################################################################
# DATA - Binary Images
######################################################################


# Generate and save random binary images.
for i in range(image_count):
    
    # Create a random array of shape (height, width) where values are 0 or 255.
    data = np.random.choice([0, 255], size=(image_size[1], image_size[0]))
    
    # Create an image from the numpy array where 'L' mode is for one-channel black-and-white images.
    image = Image.fromarray(data, 'L')
    
    # Save image.
    image.save(os.path.join(binary_output_dir, f'{i}.png'))