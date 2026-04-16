import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

import src.data_preparation as dp
import src.optimization as opt
import src.cluster_rating as cr

from tqdm import tqdm


FILE_PATH = 'data/data.csv'
OUTPUT_PATH = 'result/'

CLUSTER_COUNT = 5 # Hard-coded for now, will be made dynamic in the future
INITIALIZATION_ITERATIONS = 10 # Number of times to run the clustering process with new initial values per cluster count. The best result will be kept.

def main():
    """
    1. data prep
    2. randomly initialise centroids
    3. multiiterate optimization
    4. assign data to finalised centroids
    5. calculate iteration rating
    6. store both in seperate csv files, one for the finalised centroids and one for the iteration rating.
    7. repeat multiple times, each time different random initialised centroids
    8. store each iteration's finalised centroids separately, but the iteration rating in the same csv file by adding new rows for each iteration.
    9. determine lowest rating
    10. store in a new csv file the iterations number

    -> next upgrade will use elbow method (manual) to determine best amount of clusters as well (by repeating those ten steps for different amounts of clusters)

    -> finally export multiple csv files corresponding to playlists
    """
    data = dp.process_dataset(FILE_PATH, save_processed=True)
    initial_centroids = np.random.rand(CLUSTER_COUNT, data.shape[1]) # Seems to make most sense to scale it this way by eye as most values are between 0 and 1

    iteration_ratings = []
    for i in range(INITIALIZATION_ITERATIONS):
        print(f'Running iteration {i+1}/{INITIALIZATION_ITERATIONS}')
        final_centroids = opt.multiiterate_optimization(data.values, initial_centroids, iterations=100)
        #assigned_indices = opt.assign_points_to_centroids(data.values, final_centroids) # Not needed as it is not stored to not clutter storage. can be calculated at any point if you know the clusters, which are saved
        iteration_rating = cr.calculate_iteration_rating(data.values, final_centroids)

        np.savetxt(f'{OUTPUT_PATH}centroids/cluster{CLUSTER_COUNT}/iteration_{i}.csv', final_centroids, delimiter=',')
        iteration_ratings.append(iteration_rating)


    print(iteration_ratings)
    np.savetxt(f'{OUTPUT_PATH}iteration_ratings/cluster{CLUSTER_COUNT}/iteration_ratings.csv', iteration_ratings, delimiter=',')

main()