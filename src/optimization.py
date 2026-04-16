import numpy as np
import warnings as warnings

from tqdm import tqdm

def calculate_distances_to_point(point, subjects):
    """

    Calculates the distances between one single reference point and a set of other points (subjects)

    Parameters:
    - point: A 1D numpy bra representing the reference points' coordinates
    - subjects: An n-D numpy matrix, with each row representing the coordinates of a subject.

    Returns:
    - A 1D numpy ket containing the distances from the reference point to each subject. (rows correspond to rows in the subject matrix)

    """
    # Validate input sizes
    try:
        if point.shape[1] != subjects.shape[1]:
            raise ValueError("Point and centroids must be in the same dimensional space.")
    
        if point.shape[0] != 1:
            raise ValueError("Point must be a numpy bra (one single data point).")
    except Exception as e:
        print(f"Input validation error: {e}")
        print("Non-fatal, continuing...")


    # print("SUBJECTS")
    # print(subjects)

    # Calculate the distance from the point to each centroid
    distances = np.zeros((subjects.shape[0], 1))
    for i in range (subjects.shape[0]):
        sum_of_squared_dimensions = 0
        for j in range(point.shape[1] - 1):
            # print("FIRST THING")
            # print(point[0][j])
            # print("SECOND THING")
            # print(subjects[i][j])
            # print("SQUARED THING")
            # print((point[0][j] - subjects[i][j])**2)
            sum_of_squared_dimensions += (point[0][j] - subjects[i][j])**2
        
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
        centroid_matrices.append(np.ones((0, data_points.shape[1]))) # Initialize an empty matrix for each centroid
        for j in range(data_points.shape[0]):
            if assigned_indices[j] == i:
                centroid_matrices[i] = np.resize(centroid_matrices[i], (centroid_matrices[i].shape[0] + 1, data_points.shape[1]))
                centroid_matrices[i][centroid_matrices[i].shape[0] - 1] = (data_points[j])
    
    return centroid_matrices

def calculate_new_centroid_coordinates(assigned_points):
    """
    Parameters:
    - assigned_points: An n-D numpy matrix, with each row representing the coordinates of a data point assigned to a centroid.

    Returns:
    - A 1D numpy bra representing the new coordinates of the centroid, calculated as the mean of the assigned points.
    """
    if assigned_points.shape[0] == 0:
        warnings.warn("No points assigned to this centroid.")
    
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

def optimization_iteration(data_points, centroids, max_reseed_rounds=25):
    # """
    # Parameters:
    # - data_points: An n-D numpy matrix, with each row representing the coordinates of a data point.
    # - centroids: An n-D numpy matrix, with each row representing the coordinates of a centroid.

    # Returns:
    #     - An n-D numpy matrix, with each row representing the new coordinates of a centroid.
    # """
    # assigned_indices = assign_points_to_centroids(data_points, centroids)
    # centroid_matrices = split_into_centroid_matrices(assigned_indices, data_points, centroids)
    # new_centroids = calculate_new_centroids(centroid_matrices)
    
    # return new_centroids

    """
    Perform one k-means optimization iteration with NaN prevention via empty-cluster reseeding.

    Parameters:
    - data_points (np.ndarray): Data matrix with shape (n, d).
    - centroids (np.ndarray): Current centroid matrix with shape (k, d).
    - rng (np.random.Generator | None): RNG used to reseed empty centroids. If None, a new default
      generator is created.
    - max_reseed_rounds (int): Maximum number of reseed+reassign cycles to try while empty clusters
      exist.

    Returns:
    - np.ndarray: New centroid matrix with shape (k, d).

    Behavior:
    - If any centroid has zero assigned points, that centroid is overwritten with a random row from
      `data_points`, and assignments are recomputed. This prevents `np.mean([])` -> NaN.
    """

    rng = np.random.default_rng() # create random number generator for reseeding; this is not seeded, so it will be different on each run

    working_centroids = np.array(centroids, dtype=float, copy=True)

    centroid_matrices = None
    for _ in range(max_reseed_rounds + 1):
        assigned_indices = assign_points_to_centroids(data_points, working_centroids)
        centroid_matrices = split_into_centroid_matrices(assigned_indices, data_points, working_centroids)

        empty_idxs = [i for i, m in enumerate(centroid_matrices) if m.shape[0] == 0]
        if not empty_idxs:
            break

        # Reseed empties to random data points, then loop to reassign.
        for i in empty_idxs:
            random_row = int(rng.integers(0, data_points.shape[0]))
            working_centroids[i] = data_points[random_row]

    # Compute means; if we still have empties after repeated reseeds, keep the (reseeded) centroid
    # rather than computing mean([]) -> NaN.
    if centroid_matrices is None:
        centroid_matrices = []

    new_centroids = np.zeros_like(working_centroids)
    for i, pts in enumerate(centroid_matrices):
        if pts.shape[0] == 0:
            warnings.warn(
                "No points assigned to this centroid after reseeding; keeping centroid coordinates."
            )
            new_centroids[i] = working_centroids[i]
        else:
            new_centroids[i] = np.mean(pts, axis=0)

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
    new_centroids = optimization_iteration(data_points, centroids)

    for i in tqdm(range(iterations)):
        #print(f'Optimization iteration {i+1}/{iterations}')
        new_centroids = optimization_iteration(data_points, centroids)            
        if np.array_equal(centroids, new_centroids):
            print("Centroids have not changed, stopping optimization.")
            return new_centroids
        centroids = new_centroids

    
    return centroids