import numpy as np

def get_average_cluster_size(centroid_matrices):
    total_points = 0    
    for i in range(len(centroid_matrices)):
        total_points += centroid_matrices[i].shape[0]
    average_cluster_size = total_points / len(centroid_matrices)
    return average_cluster_size


def get_cluster_count_differences(cluster_sizes):
    differences = []
    for i in range(len(cluster_sizes) - 1):
        difference = abs(cluster_sizes[i] - cluster_sizes[i + 1])
        differences.append(difference)
    return differences

def get_elbow_point(differences):
    elbow_point = np.argmax(differences) + 1 # +1 because the differences list is one element shorter than the cluster sizes list
    return elbow_point

def elbow_method(average_sizes):
    differences = get_cluster_count_differences(average_sizes)
    elbow_point = get_elbow_point(differences)
    return elbow_point
    