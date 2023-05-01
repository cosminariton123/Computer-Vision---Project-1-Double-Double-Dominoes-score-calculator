import cv2 as cv

def Canny(image, threshold_strong_lines, threshold_weak_lines = None, sobel_kernerl_size = 3):
    if threshold_strong_lines is None:
        threshold_weak_lines = threshold_strong_lines / 2

    image = cv.Canny(image, threshold_weak_lines, threshold_strong_lines, None, apertureSize=sobel_kernerl_size)
    return image