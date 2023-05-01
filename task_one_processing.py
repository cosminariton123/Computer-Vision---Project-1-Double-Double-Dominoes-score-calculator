import numpy as np
import matplotlib.pyplot as plt
import os

from graphic_display import draw_circle, draw_line, show_image, plot_image
from feature_extraction_methodes.mid_level_feature_extraction_methodes import hough, color_segmentation, sift
from feature_extraction_methodes.low_level_feature_extraction_methodes import edge_extraction
from image_processing.colors import *
from image_processing.color_processing import get_grayscale_image, get_color_from_grayscale_image
from image_processing.image_manipulation import perspective_transformation_with_4_points, resize_image, pad_image, border_box_image, rotate_image, image_overlaying
from image_processing.filters import sharpen_image, smoothen_image_gaussian_filter, remove_noise_from_image_median_filter, dilate_image
from data_manipulation.data_loader import load_image_cv, load_video_cv
from data_manipulation.data_dumper import save_image
from image_processing.image_stitching import stitch_image_inside
from image_processing.background_extraction import extract_foreground_from_image
import cv2
    


def process_one_regular_task_image(image_path, visualize=False):
    image = cv2.imread(image_path)
    template_image = cv2.imread(os.path.join("board+dominoes", "12.jpg"))

    if visualize:

        plt.subplot(1, 3, 1)
        plot_image(template_image, title="template_image")

        plt.subplot(1, 3, 2)
        plot_image(image, title="image")

        plt.subplot(1, 3, 3)
        plot_image(image_overlaying(template_image, image), title="overlayed_with_template")

        plt.suptitle("Image not alligned with template")

        plt.show()

    image = stitch_image_inside(image, template_image, k_used_for_knn=5, ratio=0.7, ransac_rep=5)

    if visualize:
        plt.subplot(1, 3, 1)
        plot_image(template_image, title="template_image")

        plt.subplot(1, 3, 2)
        plot_image(image, title="image")

        plt.subplot(1, 3, 3)
        plot_image(image_overlaying(template_image, image), title="overlayed_with_template")

        plt.suptitle("Image alligned with template using sift features")

        plt.show()


    template_image = rotate_image(template_image, 0.12)
    template_image = template_image[170:2870, 820:3505]

    image = rotate_image(image, 0.12)
    image = image[170:2870, 820:3505]

    if visualize:
        plt.subplot(1, 2, 1)
        plot_image(template_image, title="template_image")

        plt.subplot(1, 2, 2)
        plot_image(image, title="image")

        plt.suptitle("Image and background manually vertically alligned and cropped")

        plt.show()
    

    feature_image = get_feature_image(template_image)
    if visualize:
        show_image(feature_image)
        
    h_lines = get_table_horizontal_lines(feature_image)
    v_lines = get_table_vertical_lines(feature_image)

    visualize = True


    if visualize:
        aux = template_image.copy()

        for line in v_lines:
            draw_line(aux, line[0], line[1], thickness=5)

        for line in h_lines:
            draw_line(aux, line[0], line[1], thickness=5)

        
        plt.subplot(1, 2, 1)
        plot_image(aux, title="Hough lines on template")

        aux = image.copy()

        for line in v_lines:
            draw_line(aux, line[0], line[1], thickness=5)

        for line in h_lines:
            draw_line(aux, line[0], line[1], thickness=5)

        plt.subplot(1, 2, 2)
        plot_image(aux, title="Hough lines from template overlayed on the image of interest")

        plt.suptitle("Because the perspective transform worked so well, we can compute the hough lines only on the template image")
        plt.show()

    patches = get_patches(h_lines, v_lines)

    visualize = True

    if visualize:
        aux = template_image.copy()

        for patch in patches:
            cv2.rectangle(aux, patch[0], patch[1], color=MAGENTA, thickness=5)
        
        show_image(aux)

    exit()



def get_feature_image(image):
    feature_image = get_grayscale_image(image)
    feature_image = remove_noise_from_image_median_filter(feature_image, 7)
    feature_image = edge_extraction.Canny(feature_image, 250)

    return feature_image


def get_table_horizontal_lines(feature_image):
    hough_lines = hough.get_horizontal_lines_form_image(feature_image, threshold=360)

    h_lines = list()
    for rho, theta in hough_lines:
        a = np.cos(theta)
        b = np.sin(theta)

        x0 = a*rho
        y0 = b*rho

        x1 = int(x0 + 20*(-b))
        y1 = int(y0 + 20*(a))
        x2 = int(x0 - (feature_image.shape[1] + 20) * (-b))
        y2 = int(y0 - (feature_image.shape[1] + 20) * (a))

        h_lines.append(((x1, y1), (x2, y2)))

    h_lines = hough.remove_close_lines(h_lines, 75, False)
    h_lines = h_lines[3:-1]

    return h_lines



def get_table_vertical_lines(feature_image):
    hough_lines = hough.get_lines_vertical_from_image(feature_image, threshold=360)
    
    v_lines = list()
    for rho, theta in hough_lines:
        a = np.cos(theta)
        b = np.sin(theta)

        x0 = a*rho
        y0 = b*rho

        x2 = int(x0 + (feature_image.shape[0] + 50) * (-b))
        y2 = int(y0 + (feature_image.shape[0] + 50) * (a))
        x1 = int(x0 - (20)*(-b))
        y1 = int(y0 - (20)*(a))

        v_lines.append(((x1, y1), (x2, y2)))

    v_lines = hough.remove_close_lines(v_lines, 75, True)
    v_lines = v_lines[2:-1]

    return v_lines


def get_patches(h_lines, v_lines):
    """
    Expects h_lines and v_lines sorted
    """

    def find_intersection_of_two_lines(line1, line2):
        slope_1 = (line1[1][1] - line1[0][1]) / (line1[1][0] - line1[0][0])
        slope_2 = (line2[1][1] - line2[0][1]) / (line2[1][0] - line2[0][0])

        b_1 = line1[0][1] - slope_1 * line1[0][0]
        b_2 = line2[0][1] - slope_2 * line2[0][0]

        x = (b_2 - b_1) / (slope_1 - slope_2)
        y = slope_1 * x + b_1
        return int(x), int(y)

    patch = list()
    for h_line1, h_line2 in zip(h_lines[:-1], h_lines[1:]):
        for v_line1, v_line2 in zip(v_lines[:-1], v_lines[1:]):

            point_of_intersection_upper_left_corner = find_intersection_of_two_lines(h_line1, v_line1)
            point_of_intersection_down_right_corner = find_intersection_of_two_lines(h_line2, v_line2)

            patch.append((point_of_intersection_upper_left_corner, point_of_intersection_down_right_corner))

    return patch

#TODO
def hough_circles_dots_for_later(image):
    image_gray = get_grayscale_image(image)
    circles = cv2.HoughCircles(image_gray, cv2.HOUGH_GRADIENT, 1, minDist=10, minRadius=10, maxRadius=20, param1=150, param2=30)
    circles = circles[0]

    for circle in circles:
        draw_circle(image, circle)

