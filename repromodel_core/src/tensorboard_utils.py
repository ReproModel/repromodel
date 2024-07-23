import numpy as np
import torch
from PIL import Image
from skimage.exposure import rescale_intensity

# General colormap for each label maps
colormap = (1 / 255) * np.array([
    [205, 173, 0],      # 1 yellow
    [179, 0, 0],        # 2 dark-red
    [99, 184, 255],     # 3 sky-blue
    [0, 179, 179],      # 4 green-blue
    [0, 255, 0],        # 5 green
    [173, 234, 234],    # 6 light-blue
    [255, 193, 193],    # 7 skin-color
    [255, 48, 48],      # 8 light-red
    [0, 0, 255],        # 9 blue
    [255, 140, 0],      # 10 orange
    [255, 255, 0],      # 11 yellow
    [255, 239, 213],    # 12 cream
    [255, 246, 143],    # 13 light-yellow
    [202, 255, 112],    # 14 lemon-green
    [153, 153, 255],    # 15 light-purple
    [179, 89, 0],       # 16 dark-orange
    [191, 62, 255],     # 17 strong-pink
    [171, 130, 255],    # 18 light-purple
    [230, 230, 250],    # 19 lavanda
    [250, 128, 114],    # 20 salmon
    [17, 193, 114],     # 21 green
    [250, 250, 114],    # 22 yellow
    [128, 250, 114],    # 23 light-green
    [240, 114, 120],    # 24 pink
    [0, 255, 255]       # 25 cyan
])

# Color into each label maps by using colormap
def labels2colors(labels, num_classes, sort="default"):
    colors = np.zeros((labels.shape[0], labels.shape[1], 3))
    if sort == "default":
        for i in range(num_classes):
            colors[labels == i + 1, :] = colormap[i]
    return colors

# Create Mask for image
def mask_transparency(labels, alpha):
    mask = np.zeros((labels.shape[0], labels.shape[1]))
    for i in range(25):
        if i != 0:
            mask[labels == i] = np.round(255 * alpha)
        else:
            mask[labels == i] = 255
    return mask

# Overlay label into volume 2D image 
def overlay_labels(num_classes, slice_ct, slice_labels, alpha=0.05, normalize=True, sort="default"):
    colors = labels2colors(slice_labels, num_classes, sort)
    if normalize:
        colors = rescale_intensity(colors, out_range=(0, 255))
        slice_ct = rescale_intensity(slice_ct, out_range=(0, 255))

    slice_color = np.dstack((slice_ct, slice_ct, slice_ct))
    mask = mask_transparency(slice_labels, alpha)
    
    img_masked = Image.composite(Image.fromarray(np.uint8(slice_color)),
                                 Image.fromarray(np.uint8(colors)),
                                 Image.fromarray(np.uint8(mask)))
    return img_masked

def normalize_image(image):
    """
    Normalize the image to have values between 0 and 1.
    """
    image_min = image.min()
    image_max = image.max()
    if image_min == image_max:
        return image  # Return the original image if all values are the same
    return (image - image_min) / (image_max - image_min)

def create_single_image(input):
    input_image = Image.fromarray(np.uint8(input * 255), 'L')
    return np.array(input_image.convert('RGB'))

def create_overlay_image(base_image, label_map, num_labels=4):
    overlay_image = overlay_labels(num_labels, base_image, label_map)
    return np.array(overlay_image)

def create_image(input, output):
    # Reduce tensor dimensions based on their original dimensionality
    if input.ndim == 5:  # [batch, channels, depth, height, width]
        input = input[0][0]  # Take the first sample
        output = output[0]
    elif input.ndim == 4:  # [batch, channels, height, width]
        input = input[0][0]  # Take the first sample
        output = output[0]
    else:
        raise ValueError("Unsupported tensor dimensionality")

    # Find the unique labels in the output based on number of channels
    label_map = np.argmax(output, axis=0)  # Assuming output is [channels, H, W]
    num_labels = len(np.unique(label_map))  # Number of unique labels

    base_image = input

    input_image = create_single_image(base_image)
    output_image = create_overlay_image(base_image, label_map, num_labels=num_labels)

    tensor_input_image = torch.from_numpy(input_image.copy()).permute(2, 0, 1).float() / 255.0  # Shape: [3, H, W]
    tensor_output_image = torch.from_numpy(output_image.copy()).permute(2, 0, 1).float() / 255.0  # Shape: [3, H, W]
 
    return tensor_input_image, tensor_output_image