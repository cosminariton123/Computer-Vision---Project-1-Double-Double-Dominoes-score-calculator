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

TEMPLATE_IMAGE_PATH = os.path.join("board+dominoes", "12.jpg")

def process_one_game(output_dir, game_info, visualize=False):
    image_paths, player_turns = game_info

    #-1 is the starting position, meaning that the domino is outside the track
    track_score = [-1, 1, 2, 3, 4, 5, 6, 0, 2, 5, 3, 4, 6, 2, 2, 0, 3,
                    5, 4, 1, 6, 2, 4, 5, 5, 0, 6, 3, 4, 2, 0, 1, 5,
                    1, 3, 4, 4, 4, 5, 0, 6, 3, 5, 4, 1, 3, 2, 0, 0,
                    1, 1, 2, 3, 6, 3, 5, 2, 1, 0, 6, 6, 5, 2, 1, 2,
                    5, 0, 3, 3, 5, 0, 6, 1, 4, 0, 6, 3, 5, 1, 4, 2,
                    6, 2, 3, 1, 6, 5, 6, 2, 0, 4, 0, 1, 6, 4, 4, 1,
                    6, 6, 3, 0]

    domino_board = [
                        [5, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 5],
                        [0, 0, 3, 0, 0, 4, 0, 0, 0, 4, 0, 0, 3, 0, 0],
                        [0, 3, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 3, 0],
                        [4, 0, 0, 3, 0, 2, 0, 0, 0, 2, 0, 3, 0, 0, 4],
                        [0, 0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 0, 0],
                        [0, 4, 0, 2, 0, 1, 0, 0, 0, 1, 0, 2, 0, 4, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                        [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 4, 0, 2, 0, 1, 0, 0, 0, 1, 0, 2, 0, 4, 0],
                        [0, 0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 0, 0],
                        [4, 0, 0, 3, 0, 2, 0, 0, 0, 2, 0, 3, 0, 0, 4],
                        [0, 3, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 3, 0],
                        [0, 0, 3, 0, 0, 4, 0, 0, 0, 4, 0, 0, 3, 0, 0],
                        [5, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 5]
                    ]

    rows_to_official_notation = {i : i + 1 for i in range(15)}
    collumns_to_official_notation = {0 : "A", 1 : "B", 2 : "C", 3 : "D", 4 : "E", 5 : "F", 6 : "G", 7 : "H", 8 : "I",
                                        9 : "J", 10 : "K", 11 : "L", 12 : "M", 13 : "N", 14 : "O"}

    players = {player : 0 for player in set(player_turns)}

    template_image = cv2.imread(TEMPLATE_IMAGE_PATH)
    template_image = hardcoded_rotate_and_crop(template_image)
    h_lines, v_lines, patch_matrix = get_h_lines_v_lines_patch_matrix_from_template(template_image, visualize)

    last_image_path = TEMPLATE_IMAGE_PATH
    for image_path, player_turn in zip(image_paths, player_turns):
        result = process_new_domino(image_path, last_image_path, h_lines, v_lines, patch_matrix, visualize)
        last_image_path = image_path

        first_square, second_square = result
        first_square_coordinates, first_square_dots = first_square
        second_square_coordinates, second_square_dots = second_square

        score = 0
        
        #Normal
        score += domino_board[first_square_coordinates[0]][first_square_coordinates[1]]
        score += domino_board[second_square_coordinates[0]][second_square_coordinates[1]]

        if first_square_dots == second_square_dots:
            score *= 2
        
        #bonus
        for player in players:
            if track_score[players[player]] == first_square_dots or track_score[players[player]] == second_square_dots:
                players[player] += 3
        
        players[player_turn] += score

        #Was already added in #bonus, now just incrementing for the output file
        if track_score[players[player_turn]] == first_square_dots or track_score[players[player_turn]] == second_square_dots:
            score += 3
        
        output_string = f"{rows_to_official_notation[first_square_coordinates[0]]}{collumns_to_official_notation[first_square_coordinates[1]]} {first_square_dots}\n"
        output_string += f"{rows_to_official_notation[second_square_coordinates[0]]}{collumns_to_official_notation[second_square_coordinates[1]]} {second_square_dots}\n"
        output_string += f"{score}"
        
        filename = os.path.basename(image_path).split(".")[0] + ".txt"
        with open(os.path.join(output_dir, f"{filename}"), "w") as f:
            f.write(output_string)



def process_new_domino(image_path, last_image_path, h_lines, v_lines, patches_matrix, visualize=False):
    image = cv2.imread(image_path)
    last_image = cv2.imread(last_image_path)
    template_image = cv2.imread(TEMPLATE_IMAGE_PATH)

    if visualize:

        #For image
        plt.subplot(2, 3, 1)
        plot_image(template_image, title="template_image")

        plt.subplot(2, 3, 2)
        plot_image(image, title="image")

        plt.subplot(2, 3, 3)
        plot_image(image_overlaying(template_image, image), title="image overlayed_with_template")


        #For last image
        plt.subplot(2, 3, 4)
        plot_image(template_image, title="template_image")

        plt.subplot(2, 3, 5)
        plot_image(last_image, title="last_image")

        plt.subplot(2, 3, 6)
        plot_image(image_overlaying(template_image, last_image), title="last image overlayed with template")


        plt.suptitle("Images not alligned with template")

        plt.show()

    image = stitch_image_inside(image, template_image, k_used_for_knn=5, ratio=0.7, ransac_rep=5)
    last_image = stitch_image_inside(last_image, template_image, k_used_for_knn=5, ratio=0.7, ransac_rep=5)

    if visualize:
        #For image
        plt.subplot(2, 3, 1)
        plot_image(template_image, title="template_image")

        plt.subplot(2, 3, 2)
        plot_image(image, title="image")

        plt.subplot(2, 3, 3)
        plot_image(image_overlaying(template_image, image), title="overlayed_with_template")

        #For last image
        plt.subplot(2, 3, 4)
        plot_image(template_image, title="template_image")

        plt.subplot(2, 3, 5)
        plot_image(last_image, title="last_image")

        plt.subplot(2, 3, 6)
        plot_image(image_overlaying(template_image, last_image), title="overlayed_with_template")

        plt.suptitle("Image alligned with template using sift features")

        plt.show()


    template_image = hardcoded_rotate_and_crop(template_image)
    image = hardcoded_rotate_and_crop(image)
    last_image = hardcoded_rotate_and_crop(last_image)

    if visualize:
        #For image
        plt.subplot(2, 2, 1)
        plot_image(template_image, title="template_image")

        plt.subplot(2, 2, 2)
        plot_image(image, title="image")

        #For last image
        plt.subplot(2, 2, 3)
        plot_image(template_image, title="template_image")

        plt.subplot(2, 2, 4)
        plot_image(last_image, title="last_image")

        plt.suptitle("Image and template manually vertically alligned and cropped")

        plt.show()
    

    if visualize:
        #For template
        aux = template_image.copy()

        for line in v_lines:
            draw_line(aux, line[0], line[1], thickness=5)

        for line in h_lines:
            draw_line(aux, line[0], line[1], thickness=5)

        
        plt.subplot(2, 2, 1)
        plot_image(aux, title="Hough lines on template")

        plt.subplot(2, 2, 3)
        plot_image(aux, title="Hough lines on template")

        aux = image.copy()

        for line in v_lines:
            draw_line(aux, line[0], line[1], thickness=5)

        for line in h_lines:
            draw_line(aux, line[0], line[1], thickness=5)


        #For image
        plt.subplot(2, 2, 2)
        plot_image(aux, title="Hough lines from template overlayed on the image of interest")

        #For last_image
        aux = last_image.copy()
        for line in v_lines:
            draw_line(aux, line[0], line[1], thickness=5)

        for line in h_lines:
            draw_line(aux, line[0], line[1], thickness=5)

        plt.subplot(2, 2, 4)
        plot_image(aux, title="Hough lines from template overlayed on the last_image of interest")

        plt.suptitle("Because the perspective transform worked so well, we can compute the hough lines only on the template image")
        plt.show()

    if visualize:
        aux = template_image.copy()

        for line in patches_matrix:
            for patch in line:
                cv2.rectangle(aux, patch[0], patch[1], color=MAGENTA, thickness=5)
        
        plt.subplot(2, 2, 1)
        plot_image(aux, title="Patches on template image")

        plt.subplot(2, 2, 3)
        plot_image(aux, title="Patches on template image")

        aux = image.copy()

        for line in patches_matrix:
            for patch in line:
                cv2.rectangle(aux, patch[0], patch[1], color=MAGENTA, thickness=5)

        plt.subplot(2, 2, 2)
        plot_image(aux, title="Patches from template on the image of interest")

        #For last_image
        aux = last_image.copy()

        for line in patches_matrix:
            for patch in line:
                cv2.rectangle(aux, patch[0], patch[1], color=MAGENTA, thickness=5)

        plt.subplot(2, 2, 4)
        plot_image(aux, title="Patches from template on the last_image of interest")

        plt.suptitle("Same for patches")
        plt.show()



    foreground = extract_foreground_from_image(image, last_image)
    if visualize:
        show_image(foreground, title="Difference between last image and current image")

    patches_pixels_matrix_foreground =[[get_patch_pixels(foreground, patch) for patch in line] for line in patches_matrix]
    
    maximum_difference_patch = -np.inf
    m_d_i = -1
    m_d_j = -1

    for i, line in enumerate(patches_pixels_matrix_foreground):
        for j, patch in enumerate(line):
            if np.mean(patch) > maximum_difference_patch:
                maximum_difference_patch = np.mean(patch)
                m_d_i = i
                m_d_j = j
    

    s_m_d_i = -1
    s_m_d_j = -1
    second_maximum_difference_adjacent = -np.inf

    if m_d_i > 0:
        this_patch_mean = np.mean(patches_pixels_matrix_foreground[m_d_i - 1][m_d_j])
        if this_patch_mean > second_maximum_difference_adjacent:
            second_maximum_difference_adjacent =this_patch_mean
            s_m_d_i = m_d_i - 1
            s_m_d_j = m_d_j
    if m_d_i < len(patches_pixels_matrix_foreground) - 1:
        this_patch_mean = np.mean(patches_pixels_matrix_foreground[m_d_i + 1][m_d_j])
        if this_patch_mean > second_maximum_difference_adjacent:
            second_maximum_difference_adjacent = this_patch_mean
            s_m_d_i = m_d_i + 1
            s_m_d_j = m_d_j
    if m_d_j > 0:
        this_patch_mean = np.mean(patches_pixels_matrix_foreground[m_d_i][m_d_j - 1])
        if this_patch_mean > second_maximum_difference_adjacent:
            second_maximum_difference_adjacent = this_patch_mean
            s_m_d_i = m_d_i
            s_m_d_j = m_d_j - 1
    if m_d_j < len(patches_pixels_matrix_foreground[0]) - 1:
        this_patch_mean = np.mean(patches_pixels_matrix_foreground[m_d_i][m_d_j + 1])
        if this_patch_mean > second_maximum_difference_adjacent:
            second_maximum_difference_adjacent = this_patch_mean
            s_m_d_i = m_d_i
            s_m_d_j = m_d_j + 1


    if visualize:
        aux = image.copy()

        print(f"maximum diff:{maximum_difference_patch} at {m_d_i}, {m_d_j}")
        print(f"second_maximum_diff:{second_maximum_difference_adjacent} at {s_m_d_i}, {s_m_d_j}")
        print()
        cv2.rectangle(aux, patches_matrix[m_d_i][m_d_j][0], patches_matrix[m_d_i][m_d_j][1], color=BLUE, thickness=5)
        cv2.rectangle(aux, patches_matrix[s_m_d_i][s_m_d_j][0], patches_matrix[s_m_d_i][s_m_d_j][1], color=RED, thickness=5)

        show_image(aux, title="The two sides of the domino detected")

    m_d_patch = get_patch_pixels(image, patches_matrix[m_d_i][m_d_j])
    s_m_d_patch = get_patch_pixels(image, patches_matrix[s_m_d_i][s_m_d_j])

    circles_m_d = hough_circles_get_dots(m_d_patch)
    circles_s_m_d = hough_circles_get_dots(s_m_d_patch)

    if visualize:
        for circle in circles_m_d:
            draw_circle(m_d_patch, circle)
        
        for circle in circles_s_m_d:
            draw_circle(s_m_d_patch, circle)
        
        print(f"maximum diff nr of circles: {len(circles_m_d)}")
        print(f"second_maximum_diff nr of circles: {len(circles_s_m_d)}")
        print("\n##################################\n")

        order = None
        if m_d_i < s_m_d_i or m_d_j < s_m_d_j:
            order = [1, 2]
        else:
            order = [2, 1]

        plt.subplot(abs(m_d_i - s_m_d_i) + 1, abs(m_d_j - s_m_d_j) + 1, order[0])
        plot_image(m_d_patch, title=f"Maximum diff patch, nr of circles: {len(circles_m_d)}")

        plt.subplot(abs(m_d_i - s_m_d_i) + 1, abs(m_d_j - s_m_d_j) + 1, order[1])
        plot_image(s_m_d_patch, title=f"Second max diff patch, nr of circles: {len(circles_s_m_d)}")

        plt.suptitle("Detecting domino dots using Hough Circles")
        plt.show()

    if m_d_i < s_m_d_i or m_d_j < s_m_d_j:
        return (((m_d_i, m_d_j), min(len(circles_m_d), 6)), ((s_m_d_i, s_m_d_j), min(len(circles_s_m_d), 6)))
    else:
        return (((s_m_d_i, s_m_d_j), min(len(circles_s_m_d), 6)), ((m_d_i, m_d_j), min(len(circles_m_d), 6)))


def hardcoded_rotate_and_crop(image):
    image = rotate_image(image, 0.12)
    image = image[170:2870, 820:3505]
    
    return image


def get_h_lines_v_lines_patch_matrix_from_template(template_image, visualize=False):

    feature_image = get_feature_image(template_image)
    if visualize:
        show_image(feature_image, title="Image -> Grayscale Image -> Median Filter -> Canny")

    h_lines = get_table_horizontal_lines(feature_image)
    v_lines = get_table_vertical_lines(feature_image)
    if visualize:
        aux = template_image.copy()

        for line in v_lines:
            draw_line(aux, line[0], line[1], thickness=5)

        for line in h_lines:
            draw_line(aux, line[0], line[1], thickness=5)
        
        show_image(aux, title="Vertical and horisontal hough lines with overlap removal\nand removal of other uninteresting lines")

    patches_matrix = get_patches(h_lines, v_lines)

    if visualize:
        aux = template_image.copy()

        for line in patches_matrix:
            for patch in line:
                cv2.rectangle(aux, patch[0], patch[1], color=MAGENTA, thickness=5)
        
        show_image(aux, title="Patches computed from intersection of hough lines")

    return h_lines, v_lines, patches_matrix




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

    patches_matrix = list()
    for h_line1, h_line2 in zip(h_lines[:-1], h_lines[1:]):
        line = list()
        for v_line1, v_line2 in zip(v_lines[:-1], v_lines[1:]):

            point_of_intersection_upper_left_corner = find_intersection_of_two_lines(h_line1, v_line1)
            point_of_intersection_down_right_corner = find_intersection_of_two_lines(h_line2, v_line2)

            line.append((point_of_intersection_upper_left_corner, point_of_intersection_down_right_corner))
        patches_matrix.append(line)

    return patches_matrix


def get_patch_pixels(image, patch):
    return image[patch[0][1] : patch[1][1], patch[0][0] : patch[1][0]]


def hough_circles_get_dots(patch):
    image_gray = get_grayscale_image(patch)
    circles = cv2.HoughCircles(image_gray, cv2.HOUGH_GRADIENT, 1, minDist=10, minRadius=10, maxRadius=20, param1=150, param2=30)
    
    if circles is not None:
        circles = circles[0]
        return circles

    return []
