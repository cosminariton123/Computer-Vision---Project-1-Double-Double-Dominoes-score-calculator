import cv2 as cv
import numpy as np
import os

def save_image(path, filename, image):
    cv.imwrite(os.path.join(path, filename), np.uint8(image))
