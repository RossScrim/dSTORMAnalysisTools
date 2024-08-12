from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def plot_clustered_locs(input_locs: np.ndarray, cluster_locs) -> None:
    """Visualizes DBSCAN clusters and core samples.

    This function combines the functionalities of plot_dbscan_clusters and sort_locs_by_cluster.
    It sorts locations by cluster membership, then plots the clusters using different markers 
    for core and non-core samples within each cluster.

    Args:
        input_locs (np.ndarray): Input data containing locations (x, y coordinates).
        db_result: Output of the DBSCAN algorithm run on the input locations.
    """
    
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
  
    plt.xlabel("X (nm)")
    plt.ylabel("Y (nm)")
    plt.title("Clusters")  # Adjusted title (no noise)
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

            
        # Close the hull loop by appending the first point at the end
        hull_points = np.vstack([hull_points, hull_points[0]])

        # Plot convex hull (red dashed line)
        plt.plot(hull_points[:, 0], hull_points[:, 1], 'r-', lw=2, label=f"Convex Hull (Cluster {cluster})")

    plt.xlabel("X (nm)")
    plt.ylabel("Y (nm)")
    plt.title("Clusters and Convex Hulls")
    plt.gca().invert_yaxis()  # Invert y-axis for standard convention
    plt.show()
