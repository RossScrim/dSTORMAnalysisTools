import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull


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


def create_convex_hull(cluster_localisations:dict) -> list:
    cluster_convex_hull = []
    if not cluster_localisations:
        print("No clusters provided.")
        return cluster_convex_hull

    for cluster, localisations in cluster_localisations.items():
        if not cluster == -1:
            print(f"Processing cluster {cluster} with {len(localisations)} points.")
            if len(localisations) >= 3:
                try:
                    hull = ConvexHull(localisations, incremental=False)
                    cluster_convex_hull.append(hull)
                    print(f"Convex hull created for cluster {cluster}.")
                except Exception as e:
                    print(f"Failed to create ConvexHull for cluster {cluster}: {e}")
            else:
                print(f"Cluster {cluster} skipped: insufficient points to form a ConvexHull.")

    return cluster_convex_hull


def count_localizations_per_cluster(clusters: dict) -> list:
    """
    Counts the number of localizations belonging to each cluster.

    Args:
        cluster_labels: Array of cluster assignments for each localization.

    Returns:
        list: A dictionary where keys are cluster labels (excluding -1 for noise)
                       and values are the corresponding localization counts.
    """
    number_of_locs = []
    for key, value in clusters.items():
        if key > -1:
            number_of_locs.append(len(value)) 
    return number_of_locs


def create_cluster_properties_table(cluster_data:dict, convex_hull:dict) -> pd.DataFrame:
    number_of_locs = count_localizations_per_cluster(cluster_data)
    cluster_areas = [convex_hull.volume for convex_hull in convex_hull]
    cluster_id = [cluster for cluster in list(cluster_data.keys()) if cluster >= 0]
    cluster_density = [number_of_loc / cluster_area for number_of_loc, cluster_area in zip(number_of_locs, cluster_areas)]

    data = {"cluster id" : cluster_id,   
            "number of localisations": number_of_locs,
            "Cluster Area (\u03BCm\u00B2)": cluster_areas,
            "Cluster density (locs/\u03BCm\u00B2)": cluster_density
    }
    return data