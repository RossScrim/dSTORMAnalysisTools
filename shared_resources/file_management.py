import pandas as pd
import numpy as np
import errno
import os


def get_filename(filepath):
    filepath, _ = os.path.splitext(filepath)
    filename = filepath.split("\\")
    return filename[-1]


def get_file_extension(filepath):
    _, file_extension = os.path.splitext(filepath)
    return file_extension


def generate_full_savename(filepath:str, savepath:str, save_label: str, new_file_format) -> str:
    filename = get_filename(filepath)
    savename =  f"{savepath}" + "\\" + f"{filename}_" + f"{save_label}{new_file_format}" 
    i = 1
    while os.path.exists(savename):
        savename = f"{savename[:-len(new_file_format)]}_{i}{new_file_format}"  # Add number before extension
        i += 1 
    return savename
        

def get_full_filespaths(folder_path):
    """
    This function retrieves a list of of the fullfil within a folder set for batch processing.

    rgs:
      folder_path (str): Path to the folder containing the files.

    Returns:
    list: List of full file paths within the folder.
    """
    filenames = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    full_paths = [os.path.join(folder_path, filename) for filename in filenames]
    return full_paths


def read_localisation_HDF5(filepath: str) -> pd.DataFrame:
    """Reads localization data from a HDF5 file and returns a pandas DataFrame.

    Args:
        filepath (str): Path to the HDF5 file containing localization data.

    Returns:
        pd.DataFrame: DataFrame containing localization data.
    """
    with pd.HDFStore(filepath, mode="r") as hdfstore:
        return hdfstore["locs"]
    

def read_localisation_csvdata(filepath: str) -> pd.DataFrame:
    """Reads localization data from a HDF5 file and returns a pandas DataFrame.

    Args:
        filepath (str): Path to the HDF5 file containing localization data.

    Returns:
        pd.DataFrame: DataFrame containing localization data.
    """

    return pd.read_csv(filepath)


def read_files(filepath: str) -> pd.DataFrame:
    """
    Reads localization data based on the specified format and file path.

    Args:
        file_path (str): Path to the data file.
        format (str): Expected format of the data ("csv" or "hdf5").

    Returns:
        pd.DataFrame: DataFrame containing localization data.

    Raises:
        ValueError: If the provided format is not supported.

    """
    format = get_file_extension(filepath)
    try:
     if format == ".csv":
        file = read_localisation_csvdata(filepath)
     elif format == ".hdf5":
        file = read_localisation_HDF5(filepath)
     else:
        raise ValueError("Unsupported data format:", format)
    except ValueError as e:  # Catch only ValueError (format errors)
        print(f"Skipping file '{filepath}': Unsupported format ({e})")
        file = None  # Set file to None to indicate failure
    return file


def convert_df_to_numpy(pd_locs) -> np.ndarray:
    """Converts specified columns from a pandas DataFrame to a NumPy array.

    Args:
        data (pd.DataFrame): Input pandas DataFrame.
        columns (list[int]): List of column indices to convert.

    Returns:
        np.ndarray: Converted data as a NumPy array.
    """
    return pd_locs.iloc[:,:].to_numpy()


def get_xy_loc_positions(data: np.ndarray, x: int, y:int) -> np.ndarray:
  """
  This function retrieves two specified columns from a matrix.

  Args:
      data: The input numpy array with the dstorm localisations.
      x: The index of the first column to retrieve x position.
      y: The index of the second column to retrieve y position.

  Returns:
      A new numpy array matrix containing the two specified columns.
  """
  return data[:, [x, y]]

   

if __name__ == "__main__":
    #dstorm_locs_df = read_localisation_csvdata("Y:\\Ross\\dSTORM\\ross dstorm -2.sld - cell3 - 3_locs_ROI.hdf5")
    dstorm_locs_df = read_localisation_csvdata("C:\\Users\\rscrimgeour\\Desktop\\storm_1_MMStack_Default_final_test.csv")
    X = convert_df_to_numpy(dstorm_locs_df)
    xy = get_xy_loc_positions(X, 1, 2)
    print(dstorm_locs_df)
    print(xy)
