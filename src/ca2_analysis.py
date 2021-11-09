import numpy as np
import pandas as pd
from typing import Tuple
from scipy.signal import find_peaks, butter, sosfilt
# import openpyxl


def lowPassFiltered(input_signal: np.ndarray, time_stamps: np.ndarray) -> np.ndarray:
    filter_order = 5  # how sharply the filter cuts off the larger the sharper it bends
    # frequency_range_hz = [1.0, 2.0]  # cut off frequency [start, stop] for bandpass/bandstop
    frequency_hz = 1.5
    duration = time_stamps[-1] - time_stamps[0]
    num_samples = len(time_stamps)
    sampleing_frequency = float(num_samples)/duration
    sos = butter(filter_order, frequency_hz, 'lowpass', fs=sampleing_frequency, output='sos')
    return sosfilt(sos, input_signal)


def extremePointIndices(signal: np.ndarray) -> Tuple[np.ndarray]:
    peaks, troughs = peaksAndTroughsIndices(signal)
    peaks_and_troughs = np.concatenate((peaks, troughs), axis=0)
    peaks_and_troughs_sorted = np.sort(peaks_and_troughs)
    return (peaks, troughs, peaks_and_troughs_sorted)


def peaksAndTroughsIndices(
    input_data: np.ndarray,
    time_stamps: np.ndarray=None,
    expected_frequency_hz: float=None,
    expected_min_peak_width: int=None,
    expected_min_peak_height: float=None    
) -> Tuple[np.ndarray]:
    ''' Returns the indices of peaks and troughs found in the 1D input data '''
    if expected_min_peak_height is None:
        expected_min_peak_height = 5.0
    if expected_min_peak_width is None:
        expected_min_peak_width = 5
    
    if expected_frequency_hz is not None and time_stamps is not None:
        expected_frequency_hz = expected_frequency_hz
        expected_frequency_range_hz = 0.5
        pacing_frquency_min_hz = expected_frequency_hz - expected_frequency_range_hz
        pacing_frquency_max_hz = expected_frequency_hz + expected_frequency_range_hz

        # compute the expected width (in samples) from peak to peak (or trough to trough)
        duration = time_stamps[-1] - time_stamps[0]
        num_samples = len(time_stamps)
        sampling_rate = float(num_samples)/duration
        expected_min_peak_width = sampling_rate/pacing_frquency_max_hz

        # compute the expected height from peak to peak (or trough to trough)
        # this is probably not necessary and is much harder to estimate than expected width
        # so it might cause problems with highly decaying signals and/or signals with large noise.
        # in particular, the use of half the calculated height of the middle-ish peak is entirely arbitrary
        middle_sample = int(num_samples/2)
        signal_sample_1_cycle_start = middle_sample 
        signal_sample_1_cycle_end = signal_sample_1_cycle_start + int(expected_min_peak_width)
        signal_sample_1_cycle = input_data[signal_sample_1_cycle_start:signal_sample_1_cycle_end]
        signal_sample_1_cycle_min = np.min(signal_sample_1_cycle)
        signal_sample_1_cycle_max = np.max(signal_sample_1_cycle)
        middle_peak_height = np.abs(signal_sample_1_cycle_max - signal_sample_1_cycle_min)
        min_height_scale_factor = 0.5
        expected_min_peak_height = min_height_scale_factor * middle_peak_height

    peaks, _ = find_peaks(input_data, prominence=expected_min_peak_height, distance=expected_min_peak_width)
    troughs, _ = find_peaks(-input_data, prominence=expected_min_peak_height, distance=expected_min_peak_width)
    
    return (peaks, troughs)


def ca2Data(path_to_data: str):
    ''' Reads in an xlsx file containing ca2 experiment data and
        returns a tuple of numpy arrays (time stamps, signal) '''
    ca2_data = pd.read_excel(path_to_data, usecols=[1, 5], dtype=np.float32)
    ca2_data = ca2_data.to_numpy(copy=True).T
    return (ca2_data[0], ca2_data[1])


# def extremePointIndices(signal: np.ndarray) -> Tuple[np.ndarray]:
#     from sklearn.cluster import KMeans
#     # peaks and troughs combined and sorted
#     # cluster the (edge) distance between adjacent peaks and troughs
#     # so we can eliminate the extreme points that have both edges
#     # that belong to the cluster with "small" edges (they're noise)
#     extreme_points_signal = signal[extreme_points]
#     extreme_points_edge_distances = np.abs(np.diff(extreme_points_signal))
#     extreme_points_signal = extreme_points_signal.tolist()
#     extreme_points_distance_min = np.min(extreme_points_edge_distances)
#     extreme_points_distance_max = np.max(extreme_points_edge_distances)
#     initial_cluster_centers = np.asarray([extreme_points_distance_min, extreme_points_distance_max])
#     kmeans_operator = KMeans(
#         n_clusters=2,
#         init=initial_cluster_centers.reshape(-1, 1),  # stupid sklearn requires this for 1D array
#         n_init=1
#     )
#     extreme_points_cluster_ids = kmeans_operator.fit_predict(
#         extreme_points_edge_distances.reshape(-1,1)  # stupid sklearn requires this for 1D array
#     )
#     cluster_center = kmeans_operator.cluster_centers_

#     print(cluster_center)
#     print(extreme_points_cluster_ids)
    
#     # determine which cluster contains small edge distances
#     # these are edges that contribute to bad peaks/troughs (i.e. noise)
#     if cluster_center[0] > cluster_center[1]:
#         cluster_to_remove = 0
#     else:
#         cluster_to_remove = 1
#     # remove any exteme points that have BOTH edge distances in the cluster_id_to_remove cluster
#     to_be_removed = -1
#     num_points_to_remove = 0
#     # remove points in peaks and troughs and mark points to be removed in extreme_points/_signal
#     for point_index in range(len(extreme_points_cluster_ids) - 1):
#         if extreme_points_cluster_ids[point_index] == cluster_to_remove \
#         and extreme_points_cluster_ids[point_index + 1] == cluster_to_remove:
#             index_to_remove = extreme_points[point_index + 1]
#             extreme_points[point_index + 1] = to_be_removed
#             extreme_points_signal[point_index + 1] = to_be_removed
#             num_points_to_remove += 1
#             if index_to_remove in peaks:
#                 peaks.remove(index_to_remove)
#             else:
#                 troughs.remove(index_to_remove)
#     # remove points in extreme_points/_signal that were marked for removal
#     print(f'num points to remove first round: {num_points_to_remove}')
#     for _ in range(num_points_to_remove):
#         extreme_points.remove(-1)
#         extreme_points_signal.remove(-1)
#     # now we need to look at each point and the point that follows
#     # and if there are two peaks in a row, remove the peak that is lowest, 
#     # and if there are two troughs in a row, remove the trough that is highest
#     num_points_to_remove = 0
#     for this_point_index in range(len(extreme_points) - 1):
#         next_point_index = this_point_index + 1
#         this_point = extreme_points[this_point_index]
#         next_point = extreme_points[next_point_index]
#         if this_point in peaks and next_point in peaks:
#             this_point_signal = extreme_points_signal[this_point_index]
#             next_point_signal = extreme_points_signal[next_point_index]
#             if this_point_signal < next_point_signal:
#                 peaks.remove(this_point)
#                 extreme_points[this_point_index] = to_be_removed
#                 extreme_points_signal[this_point_index] = to_be_removed
#                 num_points_to_remove += 1                
#             else:
#                 peaks.remove(next_point)
#                 extreme_points[next_point_index] = to_be_removed
#                 extreme_points_signal[next_point_index] = to_be_removed
#                 num_points_to_remove += 1
#         elif this_point in troughs and next_point in troughs:
#             this_point_signal = extreme_points_signal[this_point_index]
#             next_point_signal = extreme_points_signal[next_point_index]
#             if this_point_signal < next_point_signal:
#                 troughs.remove(this_point)
#                 extreme_points[this_point_index] = to_be_removed
#                 extreme_points_signal[this_point_index] = to_be_removed
#                 num_points_to_remove += 1
#             else:
#                 troughs.remove(next_point)
#                 extreme_points[next_point_index] = to_be_removed
#                 extreme_points_signal[next_point_index] = to_be_removed
#                 num_points_to_remove += 1
#     # remove points in extreme_points/_signal that were marked for removal
#     print(f'num points to remove second round: {num_points_to_remove}')    
#     for _ in range(num_points_to_remove):
#         extreme_points.remove(-1)
#         extreme_points_signal.remove(-1)
#     return (extreme_points, peaks, troughs)
