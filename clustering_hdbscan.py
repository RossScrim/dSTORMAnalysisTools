import pandas as pd
import os
import hdbscan
from ConfigReader import ConfigReader
import shared_resources.cluster_tools as cluster_tools
from matplotlib import pyplot as plt
from shared_resources.file_management import convert_df_to_numpy, get_full_filespaths, get_xy_loc_positions, read_files, generate_full_savename
from shared_resources.plot_functions import plot_raw_localisation_image, plot_clustered_locs, plot_convex_hull


def run_clustering_anaylsis(config, data):
    ## run DBSCAN
    eps_dis = config["DBSCAN_Paramters"]["eps"]
    min_locs = config["DBSCAN_Paramters"]["min_samples"]

    hdb = hdbscan.HDBSCAN(min_cluster_size=min_locs).fit(data)
    labels = hdb.labels_
    # Number of clusters in labels, ignoring noise if present.ALso

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)

    cluster_localisations = cluster_tools.sort_locs_by_cluster(data, hdb)
    convex_hulls = cluster_tools.create_convex_hull(cluster_localisations)

    return cluster_localisations, convex_hulls


def main():
    config = ConfigReader("ConfigFile.json").get_config()

    if config["Mode"]["Batch"]:
        os.makedirs(config["data_save_folder"], exist_ok=True)
        files_to_process = get_full_filespaths(config["data_folder_path"])
        for file in files_to_process:
            # load and convert data to type
            dstorm_locs_df = read_files(file)
            if dstorm_locs_df is not None:
                # COnvert files to numpy array and get the x,y coordinatss
                dstorm_locs_np = convert_df_to_numpy(dstorm_locs_df)
                X = get_xy_loc_positions(dstorm_locs_np, 2, 3)

                # cluster and handle
                cluster_localisations, convex_hulls = run_clustering_anaylsis(config, X)
                cluster_properties = cluster_tools.create_cluster_properties_table(cluster_localisations, convex_hulls)

                cluster_properties_df = pd.DataFrame(cluster_properties)
                save_path = generate_full_savename(file, config["data_save_folder"], "cluster_properties", ".csv")
                # save cluster_locs_files
                cluster_properties_df.to_csv(save_path, index=False, encoding="utf-8")  # Save without index column

    if config["Mode"]["Preview"]:
        dstorm_locs_df = read_files(config["preview_file_path"])
        if dstorm_locs_df is not None:
            dstorm_locs_np = convert_df_to_numpy(dstorm_locs_df)
            X = get_xy_loc_positions(dstorm_locs_np, 2, 3)

            cluster_localisations, convex_hulls = run_clustering_anaylsis(config, X)
            cluster_properties = cluster_tools.create_cluster_properties_table(cluster_localisations, convex_hulls)
            cluster_properties_df = pd.DataFrame(cluster_properties)
            print(cluster_properties_df)

            plot_raw_localisation_image(X)
            plot_clustered_locs(X, cluster_localisations)
            plot_convex_hull(cluster_localisations, convex_hulls)


if __name__ == "__main__":
    main()

