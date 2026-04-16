import numpy as np
import src.optimization as opt

def calculate_distances_of_cluster_points(data_points, centroid):
    """
    Parameters:
    - data_points: An n-D numpy matrix, with each row representing the coordinates of a data point in the cluster
    - centroid: A 1D numpy bra representing the coordinates of the centroid of the cluster.

    Returns:
    - distance_matrix: An n-D numpy matrix, with each row representing a points' distance to the centroid.
    """

    distance_matrix = opt.calculate_distances_to_point(np.reshape(centroid, (1, centroid.shape[0])), data_points)

    return distance_matrix

def calculate_mean_distance_of_cluster_points(data_points, centroid):
    """
    Parameters:
    - data_points: An n-D numpy matrix, with each row representing the coordinates of a data point in the cluster
    - centroid: A 1D numpy bra representing the coordinates of the centroid of the cluster.

    Returns:
    - mean_distance: A float representing the mean distance of the points in the cluster to the centroid.
    """

    distance_matrix = calculate_distances_of_cluster_points(data_points, centroid)
    mean_distance = np.mean(distance_matrix)

    return mean_distance

def calculate_mean_variance_to_mean_distance(mean_distance, distance_matrix):
    """
    Parameters:
    - mean_distance: A float representing the mean distance of the points in the cluster to the centroid.
    - distance_matrix: An n-D numpy matrix, with each row representing a points' distance to the centroid.
   
    Returns:
    - variance_to_mean_distance: A float representing the mean variance of the points' distances to
    """
    variances = (distance_matrix - mean_distance) ** 2
    
    variance_to_mean_distance = np.mean(variances)
    
    return variance_to_mean_distance

def calculate_cluster_rating(data_points, centroid):
    """
    Parameters:
    - data_points: An n-D numpy matrix, with each row representing the coordinates of a data point in the cluster
    - centroid: A 1D numpy bra representing the coordinates of the centroid of the cluster.

    Returns:
    - cluster_rating: A float representing the rating of the cluster, calculated as the mean variance to mean distance.
    """
    distance_matrix = calculate_distances_of_cluster_points(data_points, centroid)
    mean_distance = calculate_mean_distance_of_cluster_points(data_points, centroid)
    variance_to_mean_distance = calculate_mean_variance_to_mean_distance(mean_distance, distance_matrix)

    cluster_rating = variance_to_mean_distance 

    return cluster_rating

def calculate_iteration_rating(data_points, centroids):
    """
    Parameters:
    - data_points: An n-D numpy matrix, with each row representing the coordinates of a data point.
    - assigned_indices: A 1D numpy array, where each element represents the index of the centroid to which the corresponding data point is assigned.
    - centroids: An n-D numpy matrix, with each row representing the coordinates of a centroid.

    Returns:
    - iteration_rating: A float representing the rating of the iteration, calculated as the mean of the cluster ratings.
    """
    assigned_indices = opt.assign_points_to_centroids(data_points, centroids)
    cluster_matrices = opt.split_into_centroid_matrices(assigned_indices, data_points, centroids)
    
    cluster_ratings = []
    for i in range(len(centroids)):
        cluster_rating = calculate_cluster_rating(cluster_matrices[i], centroids[i])
        cluster_ratings.append(cluster_rating)

    
    iteration_rating = np.sum(cluster_ratings)
    

    return iteration_rating