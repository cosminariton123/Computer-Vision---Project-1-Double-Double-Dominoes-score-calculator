import cv2 as cv
from feature_extraction_methodes.mid_level_feature_extraction_methodes import sift
from image_processing.image_manipulation import get_homogeneous_matrix, perspective_transformation_with_homogeneous_matrix
from image_processing.color_processing import get_grayscale_image
import numpy as np

def match_features(features_source, features_dest, nr_of_matches_returned_from_KNN=2):
    feature_matcher = cv.DescriptorMatcher_create("FlannBased")
    matches = feature_matcher.knnMatch(features_source, features_dest, k=nr_of_matches_returned_from_KNN)   
    return matches


def stitch_image_inside(image_source, image_destination, k_used_for_knn = 5, ratio = 0.85, ransac_rep = 5):
    grayscale_image_source = get_grayscale_image(image_source)
    grayscale_image_dest = get_grayscale_image(image_destination)

    keypoints_source, features_source = sift.get_keypoints_and_features(grayscale_image_source)
    keypoints_dest, features_dest = sift.get_keypoints_and_features(grayscale_image_dest)
    all_matches = match_features(features_source, features_dest, k_used_for_knn)
    homogeneous_matrix = get_homogeneous_matrix(all_matches, keypoints_source, keypoints_dest, k_used_for_knn, ratio, ransac_rep)
    result = perspective_transformation_with_homogeneous_matrix(image_source, homogeneous_matrix)   
    mask = result[:, :] == [-1, -1, -1]
    result = result * (1 - mask) + image_destination * mask
    result = np.array(result, dtype=np.uint8)
    return result