import numpy as np
from sklearn.cluster import KMeans

def segment_image_based_on_color_KMeans(image, number_of_colors):
    assert len(image.shape) == 3 and image.shape[2] == 3, f"Image must be RGB. Given array has shape: {image.shape}"

    height, width, depth = image.shape

    image = image.reshape(height * width, depth)

    kmeans = KMeans(number_of_colors)
    labels = kmeans.fit_predict(image)
    means = kmeans.cluster_centers_
    means = [[int(elem) for elem in mean] for mean in means]

    dict_means = dict()
    for id, mean in enumerate(means):
        dict_means[id] = mean

    segmented_image = np.array([dict_means[elem] for elem in labels], dtype=np.uint8)
    segmented_image = segmented_image.reshape(height, width, depth)

    return segmented_image, means