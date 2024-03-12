import pandas as pd
import numpy as np


def read_localisation_data(filepath: str) -> pd.DataFrame:
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
