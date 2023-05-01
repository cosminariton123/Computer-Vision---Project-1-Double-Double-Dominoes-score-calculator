import matplotlib.pyplot as plt
import numpy as np
from image_processing.color_processing import get_grayscale_image

def get_differences_mask(image, background, threshold=50):
    assert image.shape == background.shape, f"Shape missmatch. Image shape is: {image.shape}. Background shape is: {background.shape}"
    assert len(image.shape) == 2 , f"Image must be grayscale. Image shape is not a 2 by 2. Given is {image.shape}"
    mask = image.astype(np.int32) - background.astype(np.int32)
    mask = np.array([[-elem if elem < 0 else elem for elem in line] for line in mask], np.uint8)
    
    mask = np.array([[0 if elem < threshold else 1 for elem in line] for line in mask], dtype=np.uint8)

    return mask


def extract_foreground_from_image(image, background, threshold = 50):
    assert image.shape == background.shape, f"Shape missmatch. Image shape is: {image.shape}. Background shape is: {background.shape}"

    if len(image.shape) == 2:
        mask = get_differences_mask(image, background, threshold)

    else:
        grayscale_image = get_grayscale_image(image)
        grayscale_background = get_grayscale_image(background)
        mask = get_differences_mask(grayscale_image, grayscale_background, threshold)
        mask = np.stack((mask, mask, mask), 2)

    result = image * mask
    result = result.astype(np.uint8)
    
    return result