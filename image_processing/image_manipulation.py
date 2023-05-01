import cv2
import numpy as np


def rotate_image(image, degrees, center="Auto", scale=1):
    height, width, _ = image.shape
    if center == "Auto":
        center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, degrees, scale)
    rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated

def translate_image(image, ox_pixels, oy_pixels):
    height, width, _ = image.shape
    affine_transformation_matrix = np.float32([[1, 0, ox_pixels],
                                            [0, 1, oy_pixels]])
    warped = cv2.warpAffine(image, affine_transformation_matrix,  (width, height))
    return warped

def affine_transformation(image, points_original_place_3, points_desired_place_3):
    height, width, _ = image.shape
    affine_transformation_matrix = cv2.getAffineTransform(points_original_place_3, points_desired_place_3)
    warped = cv2.warpAffine(image, affine_transformation_matrix, (width, height))
    return warped

def perspective_transformation_with_4_points(image, points_original_place_4, points_desired_place_4):
    points_original_place_4 = np.array(points_original_place_4, np.float32)
    points_desired_place_4 = np.array(points_desired_place_4, np.float32)

    perspective_transform_matrix = cv2.getPerspectiveTransform(points_original_place_4, points_desired_place_4)
    warped = perspective_transformation_with_homogeneous_matrix(image, perspective_transform_matrix)
    return warped

def perspective_transformation_with_homogeneous_matrix(image, homogeneous_matrix):
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
        
    warped = cv2.warpPerspective(image, homogeneous_matrix, (width, height))
    return warped

   
def get_homogeneous_matrix(all_matches, keypoints_source, keypoints_dest, k_used_for_knn = 2, ratio = 0.75, ransac_rep = 5):
    if not all_matches:
        return None
    
    matches = [] 
    for match in all_matches:  
        if len(match) == k_used_for_knn and (match[0].distance / match[1].distance) < ratio:
            matches.append(match[0])
     
    points_source = np.float32([keypoints_source[m.queryIdx].pt for m in matches])
    points_dest = np.float32([keypoints_dest[m.trainIdx].pt for m in matches])

    if len(points_source) > 4:
        homogeneous_matrix, status = cv2.findHomography(points_source, points_dest, cv2.RANSAC, ransac_rep, maxIters=10_000)
        return homogeneous_matrix
    else:
        return None


def resize_image(image, width, height, interpolation_method=cv2.INTER_CUBIC):
    if width < 10:
        _, width, _ = image.shape * width
    if height < 10:
        height, _, _ = image.shape * height
    resized_image = cv2.resize(image, (width, height), interpolation=interpolation_method)
    return resized_image

def pad_image(image, procent):
    assert len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 3, f"Image must be grayscale or RGB. Given array shape is {image.shape}."

    colored = False
    if len(image.shape) == 3:
        colored = True
        height, width, depth = image.shape
    else:
        height, width = image.shape

    padding_height = int(height * (procent / 2))
    padding_width = int(width * (procent / 2))

    if colored is True:
        padded_image = np.zeros((height + 2 * padding_height, width + 2 *padding_width, depth), dtype=np.uint8)
    else:
        padded_image = np.zeros((height + 2 * padding_height, width + 2 *padding_width), dtype=np.uint8)

    padded_image[padding_height : height + padding_height, padding_width: width + padding_width] = image.copy()
    return padded_image

def border_box_image(image, thickness_in_pixels):
    assert len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 3, f"Image must be grayscale or RGB. Given array shape is {image.shape}."

    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape

    image = image.copy()
    image[0: thickness_in_pixels] = 0
    image[height - thickness_in_pixels : height] = 0
    image[:,0: thickness_in_pixels] = 0
    image[:, width - thickness_in_pixels : width] = 0
    return image


def image_overlaying(image1_background, image2):
    return cv2.addWeighted(image1_background, 0.5, image2, 0.7, 0)
