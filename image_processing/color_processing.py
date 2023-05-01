import cv2 as cv
import numpy as np

def get_grayscale_image(image):
    image = np.array(image, np.uint8)
    return cv.cvtColor(image, cv.COLOR_RGB2GRAY)

def get_color_from_grayscale_image(image):
    image = np.array(image, np.uint8)
    return cv.cvtColor(image, cv.COLOR_GRAY2RGB)