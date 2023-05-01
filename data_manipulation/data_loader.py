import matplotlib.pyplot as plt
import cv2 as cv
import os



def load_image_cv(path, filename, grayscale=False):
    if grayscale is True:
        image = cv.imread(os.path.join(path, filename), cv.IMREAD_GRAYSCALE)
    else:
        image = cv.imread(os.path.join(path, filename))
    
    if image is None:
        raise ValueError(f"No image with name \"{filename}\"")

    return cv.cvtColor(image, cv.COLOR_BGR2RGB)


def load_video_cv(path, filename):
    video = cv.VideoCapture(os.path.join(path, filename))
    return video