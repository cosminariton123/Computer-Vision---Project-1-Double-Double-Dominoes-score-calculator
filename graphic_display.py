import cv2
import matplotlib.pyplot as plt
import numpy as np
from image_processing.colors import *

def show_image(image, grayscale=False, maximize=False, title=None):
    plot_image(image, grayscale, maximize, title)
    plt.show()


def plot_image(image, grayscale=False, maximize=False, title=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if maximize is True:
        manager = plt.get_current_fig_manager()
        manager.full_screen_toggle()

    if grayscale is True:
        plt.imshow(np.uint8(image), cmap='gray')
    else:
        plt.imshow(np.uint8(image))

    if title is not None:
        plt.title(title)


def show_image_cv2(image, image_name="image"):
    cv2.namedWindow(image_name, cv2.WINDOW_NORMAL)
    cv2.imshow(image_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def select_ROI(image):
    roi = cv2.selectROI(image)
    cv2.destroyAllWindows()
    return roi


def draw_circle(image, circle, color=MAGENTA, thickness=3):
    assert len(circle) == 3
    x, y, radius = circle
    x = int(x)
    y = int(y)
    radius = int(radius)
    cv2.circle(image, (x, y), radius, color, thickness)

def draw_line(image, start_point, end_point, color=MAGENTA, thickness=2, linetype=cv2.LINE_AA):
    x_start, y_start = start_point
    x_start = int(x_start)
    y_start = int(y_start)
    start_point = (x_start, y_start)

    x_end, y_end = end_point
    x_end = int(x_end)
    y_end = int(y_end)
    end_point = (x_end, y_end)

    cv2.line(image, start_point, end_point, color, thickness, linetype)