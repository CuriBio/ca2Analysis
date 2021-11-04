import numpy as np
import pandas as pd
from typing import Tuple
from scipy.signal import find_peaks
# import openpyxl


def ca2Data(path_to_data: str):
    ''' Reads in an xlsx file containing ca2 experiment data and
        returns a tuple of numpy arrays (time stamps, signal) '''
    ca2_data = pd.read_excel(path_to_data, usecols=[1, 5], dtype=np.float32)
    ca2_data = ca2_data.to_numpy(copy=True).T
    return (ca2_data[0], ca2_data[1])


def extremePointIndices(
    input_data: np.ndarray,
    min_height_separation: float=5.0,
    min_width_separation: int=5
) -> np.ndarray:
    ''' Returns the indices of peaks and troughs found in the 1D input data '''
    peaks, _ = find_peaks(input_data, prominence=min_height_separation, distance=min_width_separation)
    troughs, _ = find_peaks(-input_data, prominence=min_height_separation, distance=min_width_separation)
    return (peaks, troughs)