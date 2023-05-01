import cv2 as cv

def get_keypoints_and_features(image):
    assert len(image.shape) == 2, f"Image must be grayscale. Image shape is not a 2 by 2, it is {image.shape}"
    sift = cv.SIFT_create() 
    keypoints = sift.detect(image, None)
    keypoints, features = sift.compute(image, keypoints) 
    return keypoints, features