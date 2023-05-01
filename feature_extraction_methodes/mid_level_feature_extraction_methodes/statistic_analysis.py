import numpy as np


def get_center_of_mass(data):
    return np.mean(data, axis=0)


def normalize_data(data):
    return data - np.mean(data, axis=0)


def get_variance_x(data):
    data = normalize_data(data)
    return np.mean(np.array([elem**2 for elem in data[:,0]]))


def get_variance_y(data):
    data = normalize_data(data)
    return np.mean(np.array([elem**2 for elem in data[:,1]]))


def get_covariance(data):
    data = normalize_data(data)
    return np.mean(data[:,0] * data[:,1])


def get_correlation(data):
    data = normalize_data(data)
    return np.corrcoef(data[:,0], data[:,1])[0,1]
