# kmeans-spotify-clustering

A practical k-means implementation for clustering spotify songs based on attributes provided by the spotify API. The used dataset can be found [here](https://www.kaggle.com/datasets/geomack/spotifyclassification).

---
### **Overview**

#### **Purpose**
This project clusters Spotify songs using the **k-means algorithm** based on audio features (e.g., `danceability`, `energy`, `loudness`, `tempo`) from the [Spotify Songs Dataset](https://www.kaggle.com/datasets/geomack/spotifyclassification). The dataset was created using the Spotify API, but the API itself is not used in this project. The goal is to group songs by **similarity in feel and sound** to create themed playlists.

#### **Scope**
- **Input:** Spotify songs with audio features from the dataset.
- **Output:** Multiple playlists in CSV format, where each playlist contains similar songs. The playlists are **not labeled** (e.g., "high-energy" or "chill").
- **Not included:** Real-time recommendations, integration with music services (e.g., Spotify), or clustering song files (e.g., MP3).

#### **Key Features**
- **Data Extraction:** Loads and processes the dataset of Spotify songs and their audio features.
- **Preprocessing:** Cleans and normalizes the data for clustering.
- **Clustering:** Uses k-means with a multi-iteration optimization approach to group songs by similarity.
- **Elbow Method:** Automatically determines the optimal number of clusters using the elbow method.
- **Playlist Export:** Generates multiple CSV files, each containing a playlist of similar songs.

---
### **Data Pipeline**

#### **Data Source**
The project uses the **[Spotify Songs Dataset](https://www.kaggle.com/datasets/geomack/spotifyclassification)**, which includes audio features for thousands of songs.

#### **Data Preprocessing**
1. **Loading:** The dataset is loaded as a CSV file.
2. **Cleaning:**
   - Removes unused columns (`song_title`, `artist`, `target`, and an empty column) as they are irrelevant for clustering.
   - Removes the first row (which contains column names in this dataset format).
3. **Feature Selection:** All remaining columns (audio features) are used for clustering.

#### **Data Storage**
- **Processed Data:** Saved as a CSV file (`tmp/processed_data.csv`) for reuse.
- **Outputs:** Cluster centroids, iteration ratings, and playlists are stored in the `result/` directory.

---
### **K-Means Clustering**

#### **Algorithm Overview**
- **K-means** is used to group songs into clusters based on their audio features.
- The algorithm partitions the data into `k` clusters, where each song belongs to the cluster with the nearest mean (centroid).

#### **Implementation Details**
1. **Initialization:**
   - Centroids are randomly initialized for each cluster.
2. **Optimization:**
   - The algorithm iteratively refines the centroids to minimize the sum of squared distances between songs and their nearest centroid.
   - Multiple iterations are performed to avoid poor local minima.
3. **Cluster Rating:**
   - The "iteration rating" (sum of squared distances) is calculated for each run.
   - The best run (lowest rating) is selected.
4. **Elbow Method:**
   - The optimal number of clusters is determined by plotting cluster ratings against the number of clusters and selecting the "elbow" point.

---
### **Results & Postprocessing**

#### **Output**
- The clustering process generates multiple playlists, each containing similar songs.
- Each playlist is saved as a **CSV file** in the `result/` directory.

#### **Postprocessing**
- The playlists are created by grouping songs based on their assigned cluster.
- No further labeling or visualization is applied.

---
### **Code Structure**

#### **Project Layout**
```
kmeans-spotify-clustering/
├── data/
│   └── data.csv               # Original dataset
├── src/
│   ├── data_preparation.py    # Data loading, cleaning, and preprocessing
│   ├── optimization.py        # K-means clustering logic and optimization
│   ├── cluster_rating.py      # Cluster rating calculations
│   └── create_playlists.py    # Playlist export logic
├── tmp/
│   └── processed_data.csv     # Processed dataset (if saved)
├── result/
│   ├── centroids/             # Cluster centroids for each run
│   ├── iteration_ratings/     # Cluster ratings for each run
│   └── playlists.csv          # Final playlists
└── main.py                    # Main workflow and entry point
```

#### **CC Principles Applied in This Project**
- **Modularity:** Each file has a single responsibility (e.g., `data_preparation.py` handles data loading and cleaning, `optimization.py` handles clustering logic).
- **Readability:** Functions are named descriptively (e.g., `process_dataset()`, `multiiterate_optimization()`).
- **Reusability:** Processed data and intermediate results are saved for reuse.
- **Documentation:** Functions include docstrings for clarity.

---
### **Setup & Dependencies**

#### **Prerequisites**
- **[devenv](https://devenv.sh/getting-started/)** (for environment management)

#### **Installation**
1. Install devenv as per the [official guide](https://devenv.sh/getting-started/).
2. Clone the repository:
   ```bash
   git clone https://github.com/simonkdev/kmeans-spotify-clustering.git
   cd kmeans-spotify-clustering
   ```
3. Enter the devenv shell and run the project:
   ```bash
   devenv shell
   python main.py
   ```


This README was concepted by a human and written by the AI chatbot Mistral.
An AI agent by OpenAI was used for debugging minor mistakes in this project.
These mistakes did not regard the ML logic, only file and data management. 
