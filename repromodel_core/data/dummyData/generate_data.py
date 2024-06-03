import numpy as np
from PIL import Image
import os

# Create a directory to store the images if it doesn't exist
output_dir = 'repromodel_core/data/dummyData/input'
os.makedirs(output_dir, exist_ok=True)

# Define the size of the images
image_size = (256, 256)  # width, height

# Generate and save 100 random RGB images
for i in range(100):
    # Create a random array of shape (height, width, 3) and dtype uint8
    data = np.random.randint(0, 256, (image_size[1], image_size[0], 3), dtype=np.uint8)
    
    # Create an image from the numpy array
    image = Image.fromarray(data, 'RGB')
    
    # Save the image
    image.save(os.path.join(output_dir, f'{i}.png'))

print("100 random input RGB images have been created and saved.")

# Create a directory to store the images if it doesn't exist
output_dir = 'repromodel_core/data/dummyData/target'
os.makedirs(output_dir, exist_ok=True)

# Define the size of the images
image_size = (256, 256)  # width, height

# Generate and save 100 random binary images
for i in range(100):
    # Create a random array of shape (height, width) where values are 0 or 255
    data = np.random.choice([0, 255], size=(image_size[1], image_size[0]))
    
    # Create an image from the numpy array
    image = Image.fromarray(data, 'L')  # 'L' mode for one-channel black-and-white images
    
    # Save the image
    image.save(os.path.join(output_dir, f'{i}.png'))

print("100 random target binary images have been created and saved.")