import numpy as np

CLUSTER_COUNT = 5 # Hard-coded for now, will be made dynamic in the future
INITIALIZATION_ITERATIONS = 10 # Number of times to run the clustering process with new initial values per cluster count. The best result will be kept.

def calculate_distances_to_point(point, centroids):
    """
    Parameters:
    - point: A 1D numpy bra representing the data points coordinates
    - centroids: An n-D numpy matrix, with each row representing the coordinates of a centroid.

    Returns:
    - A 1D numpy ket containing the distances from the point to each centroid. (rows correspond to rows in the centroids matrix)

    """
    # Validate input sizes
    if point.shape[1] != centroids.shape[1]:
        raise ValueError("Point and centroids must be in the same dimensional space.")
    
    if point.shape[0] != 1:
        raise ValueError("Point must be a numpy bra (one single data point).")

    # Calculate the distance from the point to each centroid
    distances = np.zeros((centroids.shape[0], 1))
    for i in range (centroids.shape[0]):
        sum_of_squared_dimensions = 0
        for j in range(point.shape[1] - 1):
            sum_of_squared_dimensions += point[0][j] - centroids[i][j]**2
        distances[i][0] = np.sqrt(sum_of_squared_dimensions)

    return distances


def assign_to_closest_centroid(distances):
    """
    Parameters:
    - distances: 1D numpy ket containing the distances from a data point to each centroid, basically expects the output of the calculate_distances_to_point function.

    Returns: 
    - An integer representing the index of the closest centroid to the data point.
    """
    closest_centroid_index = np.argmin(distances)
    return closest_centroid_index


def assign_points_to_centroids(data_points, centroids):
    """
    Parameters:
    - data_points: An n-D numpy matrix, with each row representing the coordinates of a data point.
    - centroids: An n-D numpy matrix, with each row representing the coordinates of a centroid.

    Returns:
    - A 1D numpy ket containing the index of the closest centroid for each data point. (rows correspond to rows in the data_points matrix)
    """
    assigned_indices = np.zeros((data_points.shape[0], 1), dtype=int)
    for i in range(data_points.shape[0]):
        distances = calculate_distances_to_point(data_points[i:i+1], centroids)
        closest_centroid_index = assign_to_closest_centroid(distances)
        assigned_indices[i] = closest_centroid_index
    
    return assigned_indices

def split_into_centroid_matrices(assigned_indices, data_points, centroids):
    """
    Parameters:
    - assigned_indices: A 1D numpy ket containing the index of the closest centroid for each data point. (rows correspond to rows in the data_points matrix)
    - data_points: An n-D numpy matrix, with each row representing the coordinates of a data point.
    - centroids: An n-D numpy matrix, with each row representing the coordinates of a centroid.

    Returns:
    - A list of n-D numpy matrices, where each matrix contains the data points assigned to the corresponding centroid.
    """
    centroid_matrices = []
    for i in range(centroids.shape[0]):
        centroid_matrices.append(np.zeros((0, data_points.shape[1]))) # Initialize an empty matrix for each centroid
        for j in range(data_points.shape[0]):
            if assigned_indices[j] == i:
                centroid_matrices[i].append(data_points[j])
    
    return centroid_matrices

def calculate_new_centroid_coordinates(assigned_points):
    """
    Parameters:
    - assigned_points: An n-D numpy matrix, with each row representing the coordinates of a data point assigned to a centroid.

    Returns:
    - A 1D numpy bra representing the new coordinates of the centroid, calculated as the mean of the assigned points.
    """
    if assigned_points.shape[0] == 0:
        raise ValueError("No points assigned to this centroid.")
    
    new_centroid_coordinates = np.mean(assigned_points, axis=0)
    return new_centroid_coordinates

def calculate_new_centroids(centroid_matrices):
    """
    Parameters:
    - centroid_matrices: A list of n-D numpy matrices, where each matrix contains the data points assigned to the corresponding centroid.

    Returns:
    - An n-D numpy matrix, with each row representing the new coordinates of a centroid.
    """
    new_centroids = np.zeros((len(centroid_matrices), centroid_matrices[0].shape[1]))
    
    for i in range(len(centroid_matrices)):
        new_centroids[i] = calculate_new_centroid_coordinates(centroid_matrices[i])
    
    return new_centroids

def optimization_iteration(data_points, centroids):
    """
    Parameters:
    - data_points: An n-D numpy matrix, with each row representing the coordinates of a data point.
    - centroids: An n-D numpy matrix, with each row representing the coordinates of a centroid.

    Returns:
        - An n-D numpy matrix, with each row representing the new coordinates of a centroid.
    """
    assigned_indices = assign_points_to_centroids(data_points, centroids)
    centroid_matrices = split_into_centroid_matrices(assigned_indices, data_points, centroids)
    new_centroids = calculate_new_centroids(centroid_matrices)
    
    return new_centroids

def multiiterate_optimization(data_points, initial_centroids, iterations):
    """
    Parameters:
    - data_points: An n-D numpy matrix, with each row representing the coordinates of a data point.
    - initial_centroids: An n-D numpy matrix, with each row representing the initial coordinates of a centroid.
    - iterations: An integer representing the number of iterations to perform.

    Returns:
    - An n-D numpy matrix, with each row representing the final coordinates of a centroid after the specified number of iterations.
    """
    centroids = initial_centroids
    for i in range(iterations):
        centroids = optimization_iteration(data_points, centroids)
    
    return centroids