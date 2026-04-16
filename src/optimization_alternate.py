import numpy as np
import warnings as warnings

from tqdm import tqdm


def calculate_distances_to_point(point, subjects):
    """
    Calculate Euclidean distances from one reference point to a set of subjects.

    Parameters:
    - point (np.ndarray): A 2D numpy bra with shape (1, d) representing the reference point.
    - subjects (np.ndarray): An n-D numpy matrix with shape (n, d), where each row is a subject.

    Returns:
    - np.ndarray: A 2D numpy ket with shape (n, 1) containing distances from `point` to each
      row of `subjects` (row order matches `subjects`).

    Notes:
    - This intentionally mirrors the current repo implementation: it iterates over
      `range(d - 1)` (i.e., ignores the last dimension).
    """
    # Validate input sizes (non-fatal; matches current repo behavior)
    try:
        if point.shape[1] != subjects.shape[1]:
            raise ValueError("Point and centroids must be in the same dimensional space.")

        if point.shape[0] != 1:
            raise ValueError("Point must be a numpy bra (one single data point).")
    except Exception as e:
        print(f"Input validation error: {e}")
        print("Non-fatal, continuing...")

    # Calculate the distance from the point to each subject
    distances = np.zeros((subjects.shape[0], 1))
    for i in range(subjects.shape[0]):
        sum_of_squared_dimensions = 0
        # NOTE: This intentionally mirrors src/optimization.py (it ignores the last column).
        for j in range(point.shape[1] - 1):
            sum_of_squared_dimensions += (point[0][j] - subjects[i][j]) ** 2
        distances[i][0] = np.sqrt(sum_of_squared_dimensions)

    return distances


def assign_to_closest_centroid(distances):
    """
    Parameters:
    - distances (np.ndarray): A 2D numpy ket with shape (k, 1), distances to each centroid.

    Returns:
    - int: Index of the closest centroid (argmin over `distances`).
    """
    return np.argmin(distances)


def assign_points_to_centroids(data_points, centroids):
    """
    Parameters:
    - data_points (np.ndarray): An n-D numpy matrix with shape (n, d), each row a data point.
    - centroids (np.ndarray): An n-D numpy matrix with shape (k, d), each row a centroid.

    Returns:
    - np.ndarray: A 2D numpy ket with shape (n, 1) containing the centroid index assigned to
      each data point (row order matches `data_points`).
    """
    assigned_indices = np.zeros((data_points.shape[0], 1), dtype=int)
    for i in range(data_points.shape[0]):
        distances = calculate_distances_to_point(data_points[i : i + 1], centroids)
        assigned_indices[i] = assign_to_closest_centroid(distances)
    return assigned_indices


def split_into_centroid_matrices(assigned_indices, data_points, centroids):
    """
    Parameters:
    - assigned_indices (np.ndarray): A 2D numpy ket with shape (n, 1), centroid index per point.
    - data_points (np.ndarray): An n-D numpy matrix with shape (n, d), each row a data point.
    - centroids (np.ndarray): An n-D numpy matrix with shape (k, d), used for k and dimensionality.

    Returns:
    - list[np.ndarray]: Length-k list of matrices; the i-th matrix has shape (n_i, d) and contains
      the points assigned to centroid i.
    """
    centroid_matrices = []
    for i in range(centroids.shape[0]):
        centroid_matrices.append(np.ones((0, data_points.shape[1])))
        for j in range(data_points.shape[0]):
            if assigned_indices[j] == i:
                centroid_matrices[i] = np.resize(
                    centroid_matrices[i],
                    (centroid_matrices[i].shape[0] + 1, data_points.shape[1]),
                )
                centroid_matrices[i][centroid_matrices[i].shape[0] - 1] = data_points[j]
    return centroid_matrices


def calculate_new_centroid_coordinates(assigned_points):
    """
    Parameters:
    - assigned_points (np.ndarray): Matrix with shape (n_i, d) of points assigned to one centroid.

    Returns:
    - np.ndarray: A 1D numpy bra with shape (d,) representing the mean of `assigned_points`.
    """
    return np.mean(assigned_points, axis=0)


def calculate_new_centroids(centroid_matrices):
    """
    Parameters:
    - centroid_matrices (list[np.ndarray]): Length-k list; i-th entry has shape (n_i, d).

    Returns:
    - np.ndarray: An n-D numpy matrix with shape (k, d) of updated centroid coordinates.
    """
    new_centroids = np.zeros((len(centroid_matrices), centroid_matrices[0].shape[1]))
    for i in range(len(centroid_matrices)):
        new_centroids[i] = calculate_new_centroid_coordinates(centroid_matrices[i])
    return new_centroids


def optimization_iteration(data_points, centroids, max_reseed_rounds=25):
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


def multiiterate_optimization(data_points, initial_centroids, iterations, rng=None):
    """
    Run repeated optimization iterations with NaN-safe empty-cluster reseeding.

    Parameters:
    - data_points (np.ndarray): Data matrix with shape (n, d).
    - initial_centroids (np.ndarray): Initial centroid matrix with shape (k, d).
    - iterations (int): Max number of optimization iterations to run.
    - rng (np.random.Generator | None): RNG forwarded to `optimization_iteration`. If None, a new
      default generator is created.

    Returns:
    - np.ndarray: Final centroid matrix with shape (k, d).
    """
    if rng is None:
        rng = np.random.default_rng()

    centroids = initial_centroids
    new_centroids = optimization_iteration(data_points, centroids, rng=rng)

    for _ in tqdm(range(iterations)):
        new_centroids = optimization_iteration(data_points, centroids, rng=rng)
        if np.array_equal(centroids, new_centroids):
            print("Centroids have not changed, stopping optimization.")
            return new_centroids
        centroids = new_centroids

    return centroids
