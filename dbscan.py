import numpy as np
import pandas as pd
import collections 
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from sklearn import metrics
from sklearn.cluster import DBSCAN
from shared_resources.shared_resources import convert_df_to_numpy, get_files, get_xy_loc_positions, read_files
from ConfigReader import ConfigReader



def sort_locs_by_cluster(input_locs, db_result):
    """
    This function combines cluster membership and core sample information to identify core points
    belonging to each cluster. It creates a dictionary where keys are unique cluster labels and
    values are lists of unique core point locations (x, y coordinates) for each cluster.

    Args:
        input_locs (np.ndarray): Input data containing locations (x, y coordinates).
        db_result: Output of the DBSCAN algorithm run on the input locations.

    Returns:
        dict: A dictionary mapping unique cluster labels to lists of unique core point locations.
    """
    labels = db_result.labels_
    unique_labels = set(labels)

    cluster_locs = {}  # Use a regular dictionary (no default value needed)
    for k in unique_labels:
        cluster_locs[k] = []  # Initialize an empty list for the cluster
        class_member_mask = labels == k
        cluster_locs[k] = (input_locs[class_member_mask])

    return cluster_locs

        
def count_localizations_per_cluster(cluster_labels: np.ndarray) -> dict[int, int]:
    """
    Counts the number of localizations belonging to each cluster.

    Args:
        cluster_labels: Array of cluster assignments for each localization.

    Returns:
        dict[int, int]: A dictionary where keys are cluster labels (excluding -1 for noise)
                       and values are the corresponding localization counts.
    """

    unique_labels, counts = np.unique(cluster_labels[cluster_labels >= 0], return_counts=True)
    return dict(zip(unique_labels, counts))


def create_convex_hull(cluster_localisations:dict) -> list:
    if cluster_localisations:
        cluster_convex_hull = []
        for cluster, localisations in cluster_localisations.items():
            if cluster >= 0:
                cluster_convex_hull.append(ConvexHull(localisations, incremental=True))
    return cluster_convex_hull 


def count_localizations_per_cluster(cluster_labels: np.ndarray) -> dict[int, int]:
    """
    Counts the number of localizations belonging to each cluster.

    Args:
        cluster_labels: Array of cluster assignments for each localization.

    Returns:
        dict[int, int]: A dictionary where keys are cluster labels (excluding -1 for noise)
                       and values are the corresponding localization counts.
    """

    unique_labels, counts = np.unique(cluster_labels[cluster_labels >= 0], return_counts=True)
    return dict(zip(unique_labels, counts))


def create_cluster_properties_table():
    pass


def plot_clustered_locs(input_locs: np.ndarray, db_result) -> None:
    """Visualizes DBSCAN clusters and core samples.

    This function combines the functionalities of plot_dbscan_clusters and sort_locs_by_cluster.
    It sorts locations by cluster membership, then plots the clusters using different markers 
    for core and non-core samples within each cluster.

    Args:
        input_locs (np.ndarray): Input data containing locations (x, y coordinates).
        db_result: Output of the DBSCAN algorithm run on the input locations.
    """
    
    cluster_locs = sort_locs_by_cluster(input_locs, db_result)  # Get dictionary of core points per cluster
    plt.figure(2)
    # Define a colormap for clusters (excluding noise)
    cmap = plt.cm.get_cmap("tab10", len(cluster_locs))  # Adjust length for number of clusters

    # Define markers for core and noise points (not used here, noise points excluded)
    core_marker = "o"  # Not used in this version

    for cluster, locs in cluster_locs.items():
        if cluster == -1:
            # Use the cluster label (integer) as the index for the colormap
            plt.plot(locs[:, 0], locs[:, 1], "x", markerfacecolor=cmap(cluster), markeredgecolor='k', markersize=4, label=f"Noise")
        else:
            # Use the cluster label (integer) as the index for the colormap
            plt.plot(locs[:, 0], locs[:, 1], core_marker, markerfacecolor=cmap(cluster), markeredgecolor='k', markersize=4, label=f"Cluster {cluster}")
  
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.title("Clusters")  # Adjusted title (no noise)
    plt.legend()
    plt.gca().invert_yaxis()  # Invert y-axis for standard convention
    plt.show(block=False)


def plot_raw_localisation_image(data: np.ndarray, marker_size: float = 1) -> None:
    """Plots a scatterplot of the raw localization data.

    Args:
        data (np.ndarray): Input data as a NumPy array.
        markersize (float, optional): Size of the markers in the scatterplot. Defaults to 0.9.
    """
    plt.figure(1)
    plt.scatter(data[:, 0], data[:, 1], marker=".", s=marker_size)
    plt.gca().invert_yaxis()  # Invert y-axis for standard convention
    plt.show(block=False)


def plot_convex_hull(cluster_localisations:dict, convex_hulls) -> None:
    # Define a colormap for clusters (excluding noise)
    plt.figure(3)
    cmap = plt.cm.get_cmap("tab10", len(cluster_localisations))  # Adjust length for number of clusters

    # Loop through clusters and plot points and convex hulls
    for cluster, locs in cluster_localisations.items():
        if cluster == -1:
            continue

        # Plot cluster points
        plt.plot(locs[:, 0], locs[:, 1], 'o', markerfacecolor=cmap(cluster), markeredgecolor='k', markersize=4, alpha=0.7, label=f"Cluster {cluster}")

        # Get convex hull for this cluster
        convex_hull = convex_hulls[cluster]  # Assuming cluster numbering starts from 1 (modify if needed)
        hull_points = locs[convex_hull.vertices]  # Extract hull points

        # Plot convex hull (red dashed line)
        plt.plot(hull_points[:, 0], hull_points[:, 1], 'r-', lw=2, label=f"Convex Hull (Cluster {cluster})")

    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.title("Clusters and Convex Hulls")
    plt.legend()
    plt.gca().invert_yaxis()  # Invert y-axis for standard convention
    plt.show()

    
def run_clustering_anaylsis(config, data):
    ## run DBSCAN
    eps = config["DBSCAN_Paramters"]["eps"]
    min_samples = config["DBSCAN_Paramters"]["eps"]

    db = DBSCAN(eps, min_samples).fit(data)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.ALso 
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    
    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)

    cluster_localisations = sort_locs_by_cluster(data, db)
    convex_hulls = create_convex_hull(cluster_localisations)

    plot_clustered_locs(X, db)
    plot_convex_hull(cluster_localisations, convex_hulls)

    return cluster_localisations, convex_hulls

def create_data_table():
    pass

def main():
    
    config = ConfigReader("ConfigFile.json").get_config()
    
    if config["Batch_processing"]:
        files_to_process = get_files(config["data_folder_path"])
        for file in files_to_process:
            dstorm_locs_df = read_files(file, format=".csv")
            dstorm_locs_np = convert_df_to_numpy(dstorm_locs_df)
            X = get_xy_loc_positions(dstorm_locs_np, 1, 2)
            cluster_localisations, convex_hulls = run_clustering_anaylsis(config, X)

            

    else:
        
        ## LOAD HDF5 LOCALISATION FILES
        dstorm_locs_df = dead_files(file, format=".csv")
    
        ## LOAD CSV LOCALISATION FILES
        #dstorm_locs_df = read_localisation_csvdata("C:\\Users\\rscrimgeour\\Desktop\\storm_1_MMStack_Default_final_test.csv")
        
        dstorm_locs_np = convert_df_to_numpy(dstorm_locs_df)
        X = get_xy_loc_positions(dstorm_locs_np, 1, 2)
        
        run_clustering(config, X)

if __name__ == "__main__":
    main()

