import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms


def ImageLoader(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transforms.ToTensor()(image).unsqueeze(0)
    return image


def resize_image(image, max_size=720):
    """
    Resize an image to have a maximum dimension of `max_size` while maintaining aspect ratio.
    """
    width, height = image.size
    if max(width, height) > max_size:
        if width > height:
            new_width = max_size
            new_height = int(new_width * height / width)
        else:
            new_height = max_size
            new_width = int(new_height * width / height)
        return image.resize((new_width, new_height), Image.LANCZOS)
    return image


def ImageShow(tensor, title, size=(10, 8), save=False):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    if save:
        image.save(title + ".jpg")
    plt.figure(figsize=size)
    plt.imshow(image)
    plt.title(title)
    plt.show()


def device_checker(device: str = "cuda:0"):
    if "cuda" in device:
        if torch.cuda.is_available():
            return device
    return "cpu"


def max_column_by_category(np_mask):
    # Extract the separate color channels
    red_channel = np_mask[:, :, 0]
    green_channel = np_mask[:, :, 1]
    blue_channel = np_mask[:, :, 2]

    # Find the maximum value for each channel
    max_red = red_channel.max()
    max_green = green_channel.max()
    max_blue = blue_channel.max()
    return max_red, max_green, max_blue


# Convert RGB to grayscale using the weighted sum formula
def rgb_to_grayscale(np_mask):
    # Extract the RGB channels
    red_channel = np_mask[:, :, 0]
    green_channel = np_mask[:, :, 1]
    blue_channel = np_mask[:, :, 2]

    # Calculate the grayscale values using the weights
    grayscale = 0.299 * red_channel + 0.587 * green_channel + 0.114 * blue_channel

    # Convert to uint8 type and return
    return grayscale.astype(np.uint8)

