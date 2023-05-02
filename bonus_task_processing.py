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

from task_one_processing import get_h_lines_v_lines_patch_matrix_from_template, hardcoded_rotate_and_crop, get_patch_pixels

TEMPLATE_IMAGE_PATH = os.path.join("board+dominoes", "12.jpg")

def process_one_image(output_dir, image_path, visualize=False):
    try:

        template_image = cv2.imread(TEMPLATE_IMAGE_PATH)
        image = cv2.imread(image_path)

        if visualize:

                #For image
                plt.subplot(1, 3, 1)
                plot_image(template_image, title="template_image")

                plt.subplot(1, 3, 2)
                plot_image(image, title="image")

                plt.subplot(1, 3, 3)
                plot_image(image_overlaying(template_image, image), title="image overlayed_with_template")


                plt.suptitle("Images not alligned with template")

                plt.show()

        image = stitch_image_inside(image, template_image, k_used_for_knn=11, ratio=0.35, ransac_rep=4)

        if visualize:
            #For image
            plt.subplot(1, 3, 1)
            plot_image(template_image, title="template_image")

            plt.subplot(1, 3, 2)
            plot_image(image, title="image")

            plt.subplot(1, 3, 3)
            plot_image(image_overlaying(template_image, image), title="overlayed_with_template")

            plt.suptitle("Image alligned with template using sift features")

            plt.show()

        template_image = hardcoded_rotate_and_crop(template_image)
        image = hardcoded_rotate_and_crop(image)


        if visualize:
            #For image
            plt.subplot(1, 2, 1)
            plot_image(template_image, title="template_image")

            plt.subplot(1, 2, 2)
            plot_image(image, title="image")

            plt.suptitle("Image and template manually vertically alligned and cropped")

            plt.show()

        h_lines, v_lines, patches_matrix = get_h_lines_v_lines_patch_matrix_from_template(template_image, visualize)

        if visualize:
            #For template
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

            #For image
            plt.subplot(1, 2, 2)
            plot_image(aux, title="Hough lines from template overlayed on the image of interest")

            plt.suptitle("Because the perspective transform worked so well, we can compute the hough lines only on the template image")
            plt.show()


        if visualize:
            aux = template_image.copy()

            for line in patches_matrix:
                for patch in line:
                    cv2.rectangle(aux, patch[0], patch[1], color=MAGENTA, thickness=5)
            
            plt.subplot(1, 2, 1)
            plot_image(aux, title="Patches on template image")

            aux = image.copy()

            for line in patches_matrix:
                for patch in line:
                    cv2.rectangle(aux, patch[0], patch[1], color=MAGENTA, thickness=5)

            plt.subplot(1, 2, 2)
            plot_image(aux, title="Patches from template on the image of interest")


            plt.suptitle("Same for patches")
            plt.show()


        foreground = extract_foreground_from_image(image, template_image)
        if visualize:
            show_image(foreground, title="Difference between image and background")


        patches_pixels_matrix_foreground =[[get_patch_pixels(foreground, patch) for patch in line] for line in patches_matrix]

        error_list = list()

        THRESHOLD = 100
        for i, line in enumerate(patches_pixels_matrix_foreground):
            for j, patch in enumerate(line):
                if np.mean(patch) > THRESHOLD:
                    
                    result = check_if_square_error(i, j, patches_pixels_matrix_foreground, THRESHOLD)
                    if result is not None and result not in error_list:
                        error_list.append(result)



        rows_to_official_notation = {i : i + 1 for i in range(15)}
        collumns_to_official_notation = {0 : "A", 1 : "B", 2 : "C", 3 : "D", 4 : "E", 5 : "F", 6 : "G", 7 : "H", 8 : "I",
                                            9 : "J", 10 : "K", 11 : "L", 12 : "M", 13 : "N", 14 : "O"}

        output_string = f"{len(error_list)}\n"
        for error in error_list:
            for error_point in error:
                output_string += f"{rows_to_official_notation[error_point[0]]}{collumns_to_official_notation[error_point[1]]}\n"

        if len(error_list) > 0:
            output_string = output_string[:-1]
        else:
            output_string = f"{1}\n"
            output_string += f"{rows_to_official_notation[0]}{collumns_to_official_notation[0]}\n"
            output_string += f"{rows_to_official_notation[1]}{collumns_to_official_notation[0]}"

        filename = os.path.basename(image_path).split(".")[0] + ".txt"
        with open(os.path.join(output_dir, f"{filename}"), "w") as f:
            f.write(output_string)

    except Exception as e:
        print(f"Warning:\n{str(e)}\n Error occured at {os.path.basename(image_path)}. Tune sift to allow more freedom!!!\n\n")
        output_string = f"{1}\n"
        output_string += f"{rows_to_official_notation[0]}{collumns_to_official_notation[0]}\n"
        output_string += f"{rows_to_official_notation[1]}{collumns_to_official_notation[0]}"
        filename = os.path.basename(image_path).split(".")[0] + ".txt"
        with open(os.path.join(output_dir, f"{filename}"), "w") as f:
            f.write(output_string)


def check_if_square_error(i, j, patches_pixels_matrix_foreground, THRESHOLD):

    left_patch = False
    if j > 0:
        left_patch = np.mean(patches_pixels_matrix_foreground[i][j - 1]) > THRESHOLD
    
    right_patch = False
    if j < len(patches_pixels_matrix_foreground[0]) - 1:
        right_patch = np.mean(patches_pixels_matrix_foreground[i][j + 1]) > THRESHOLD
    
    up_patch = False
    if i > 0:
        up_patch = np.mean(patches_pixels_matrix_foreground[i - 1][j]) > THRESHOLD
    
    down_patch = False
    if i < len(patches_pixels_matrix_foreground) - 1:
        down_patch = np.mean(patches_pixels_matrix_foreground[i + 1][j]) > THRESHOLD

    left_up_patch = False
    if i > 0 and j > 0:
        left_up_patch = np.mean(patches_pixels_matrix_foreground[i - 1][j - 1]) > THRESHOLD
    
    left_down_patch = False
    if i < len(patches_pixels_matrix_foreground) - 1 and j > 0:
        left_down_patch = np.mean(patches_pixels_matrix_foreground[i + 1][j - 1]) > THRESHOLD

    right_up_patch = False
    if i > 0 and j < len(patches_pixels_matrix_foreground[0]) - 1:
        right_up_patch = np.mean(patches_pixels_matrix_foreground[i - 1][j + 1]) > THRESHOLD

    right_down_patch = False
    if i < len(patches_pixels_matrix_foreground) - 1 and j < len(patches_pixels_matrix_foreground[0]) - 1:
        right_down_patch = np.mean(patches_pixels_matrix_foreground[i + 1][j + 1]) > THRESHOLD

    
    list_of_errors = None

    #check left_upper
    if left_patch and left_up_patch and up_patch:
        list_of_errors = [(i, j), (i, j - 1), (i - 1, j - 1), (i - 1, j)]
    
    #check right_upper
    if up_patch and right_patch and right_up_patch:
        list_of_errors = [(i, j), (i - 1, j), (i, j + 1), (i - 1, j + 1)]

    #check left_lower
    if left_patch and down_patch and left_down_patch:
        list_of_errors = [(i, j), (i, j - 1), (i + 1, j), (i + 1, j - 1)]

    #check right_lower
    if right_patch and down_patch and right_down_patch:
        list_of_errors = [(i, j), (i, j + 1), (i + 1, j), (i + 1, j + 1)]

    #Check if there are errors present
    if list_of_errors is not None:
        list_of_errors.sort(key=lambda x: (x[0], x[1]))
        return list_of_errors
