import src.optimization as opt
import src.data_preparation as dp


import numpy as np
import pandas as pd
import tqdm

CHOSEN_ITERATION_INDEX = 0

def pull_clusters(data_points, iteration_index=CHOSEN_ITERATION_INDEX):
    
    centroids = pd.read_csv(
        f"result/centroids/cluster5/iteration_{iteration_index}.csv",
        header=None,
    ).to_numpy()
    assigned_indices = opt.assign_points_to_centroids(data_points, centroids)
    centroid_matrices = opt.split_into_centroid_matrices(assigned_indices, data_points, centroids)
    
    return centroid_matrices

def export_csv_playlists(centroid_matrices):

    for i in range(len(centroid_matrices)):
        np.savetxt(
            f"result/playlists/playlist_{i}.csv",
            centroid_matrices[i],
            delimiter=",",
            fmt="%s",
        )

    return 0

def find_song_index(data_points, song_point):
    matching_indices = data_points[data_points.apply(lambda row: np.array_equal(row.values, song_point), axis=1)].index

    # if len(matching_indices) > 0:
    #     print("Row index:", matching_indices[0])
    # else:
    #     print("No matching row found.")

    return matching_indices[0] if len(matching_indices) > 0 else None

def create_playlists(data_points, iteration_index=CHOSEN_ITERATION_INDEX):

    print(type(data_points.to_numpy()))
    centroid_matrices = pull_clusters(data_points.to_numpy(), iteration_index)

    raw_data = pd.read_csv("data/data.csv").to_numpy()
    playlists = []

    print("Creating playlists...")

    for i in tqdm.tqdm(range(len(centroid_matrices))):
        playlists.append([])
        print(f"Playlist {i}:")
        for song_point in centroid_matrices[i]:
            song_index = find_song_index(data_points, song_point)
            if song_index is not None:
                song_info = raw_data[song_index][15:]
                playlists[i].append(song_info)
            else:
                print("No matching song found for point:", song_point)

    return playlists

def playlist_api(data_points, iteration_index=CHOSEN_ITERATION_INDEX, cluster_count=5):
    
    playlists = create_playlists(data_points, iteration_index)
    
    export_csv_playlists(playlists)
    
    return playlists
