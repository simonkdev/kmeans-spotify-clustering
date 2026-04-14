import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

import src.data_preparation as dp
import src.optimization as opt
import src.cluster_rating as cr


FILE_PATH = 'data/data.csv'

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
    