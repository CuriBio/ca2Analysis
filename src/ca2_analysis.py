import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from scipy.signal import find_peaks, butter, sosfilt
from numpy.polynomial.polynomial import polyfit, Polynomial
# import openpyxl


# TODO: check there are no double peaks or troughs.
#       would need to use the peak and trough indicies
#       find if the peak or trough is first
#       then oscilate between peaks and troughs ensuring
#       the values in time_stamps of the alternating sequence
#       of peak and trough indicies forms a strictly monotonically increase sequence
#       i.e. from one to the next the time is always greater
#       because if there is ever a double peak, the alternating sequence wont be monotonically increasing
#       the alterniative is to just throw a tuple ('peak or trough', index) into a list
#       and sort on the index then step through the list and ensure we alternate

# TODO: IF the current method of estimate the various metrics where we
#       use a polynomial fit and then find the roots of the poly FAILS,
#       numpy warnings such as the rank is blah or the fit is ill-conditioned etc ... 
#       we will have to use a method like:
#       take the polynomial coefficients and perform a search of the function
#       between the start and end times
#       could do this with a bisection and stop when the step size < some tol in seconds say 0.001
#       or we could do a multi grid search i.e step size of 0.1 second
#       then 0.01 second between the points we narrowed it to
#       then 0.001 between the subsequent points we narrow it to, etc
#       if the fit still fails, we'd have to find the two closest time points that the
#       metric parameter lives between, and linear iteropolate between those two points
#       to find the point the metric is for. this would clearly be inferiour since
#       we know most of these signals have exponential or polynomial shape and
#       a linear interpolation between two points (even if close) isn't a great fit.
#       or, 
#       the alternative is to use splines which do essentially the same thing
#       without having to explicitly code the linear interpolation, but since we have to 
#       compute enough splines to for the resolution we need computationally very expensive


def pointToPointMetrics(
    start_point_indices: np.ndarray,
    end_point_indices: np.ndarray,
    point_values: np.ndarray,
    point_times: np.ndarray
) -> Dict:
    endpoint_value_fractions = np.asarray([0.5, 0.9], dtype=np.float32)
    num_metrics_to_compute = len(endpoint_value_fractions) + 1  # +1 for 100% case
    num_point_values = len(start_point_indices)
    metrics = np.zeros(shape=(num_metrics_to_compute, num_point_values), dtype=np.float32)
    metric_failure_counter = np.zeros(shape=(num_metrics_to_compute), dtype=np.float32)

    for point_index in range(len(start_point_indices)):
        start_point_index = start_point_indices[point_index]
        end_point_index = end_point_indices[point_index] + 1
        start_time = point_times[start_point_index]
        points_to_fit_poly_at = point_times[start_point_index:end_point_index] - start_time  # shift times to 0
        value_of_points_to_fit = point_values[start_point_index:end_point_index]

        start_time = points_to_fit_poly_at[0]
        end_time = points_to_fit_poly_at[-1]
        metrics[-1, point_index] = end_time
        
        num_points_for_fit = len(points_to_fit_poly_at)
        if num_points_for_fit > 5:
            polyfit_deg = 3
        else:
            polyfit_deg = 2

        polyfit_of_values = Polynomial.fit(
            points_to_fit_poly_at,
            value_of_points_to_fit,
            polyfit_deg,
            domain=[start_time, end_time],
            window=[start_time, end_time]
        )
        poly = Polynomial(polyfit_of_values.convert().coef)
        
        start_point_time = points_to_fit_poly_at[0]
        end_point_time = points_to_fit_poly_at[-1]
        start_point_value = value_of_points_to_fit[0]
        end_point_value = value_of_points_to_fit[-1]
        point_to_point_value = end_point_value - start_point_value
        for fraction_id_to_add in range(len(endpoint_value_fractions)):
            fraction_of_value = endpoint_value_fractions[fraction_id_to_add]
            fractional_value = start_point_value + fraction_of_value*point_to_point_value
            roots = Polynomial.roots(poly - fractional_value)
            failure_count = 1.0
            for root in roots:
                if np.iscomplex(root):
                    continue  # only true if imaginary part is non zero (could still be a complex num object) 
                if root < start_point_time or root > end_point_time:
                    continue
                metrics[fraction_id_to_add, point_index] = np.real(root)  # could be complex num obj with imaginary part 0
                failure_count = 0.0
                break
            metric_failure_counter[fraction_id_to_add] += failure_count

    metrics_counters = np.abs(metric_failure_counter - num_point_values)
    metrics_sums = np.sum(metrics, axis=-1)
    metrics_means = metrics_sums/metrics_counters
    metrics_failure_proportions = metric_failure_counter/num_point_values
    return {
        'p2p_metric_data': metrics,
        'mean_metric_data': metrics_means,
        'metric_failure_proportions': metrics_failure_proportions
    }


def ca2Metrics(
    value_data: np.ndarray,
    time_stamps: np.ndarray=None,
    expected_frequency_hz: float=None,
    expected_min_peak_width: int=None,
    expected_min_peak_height: float=None    
) -> Tuple[Dict]:

    peak_indices, trough_indices = peakAndTroughIndices(
        value_data,
        time_stamps,
        expected_frequency_hz=expected_frequency_hz,
        expected_min_peak_width=expected_min_peak_width,
        expected_min_peak_height=expected_min_peak_height
    )

    first_peak_time = time_stamps[peak_indices[0]]
    first_trough_time = time_stamps[trough_indices[0]]

    # compute the trough to peak metrics
    trough_sequence_start = 0
    if first_trough_time < first_peak_time:
        peak_sequence_start = 0
    else:
        peak_sequence_start = 1
    num_troughs = len(trough_indices)
    num_useable_peaks = len(peak_indices) - peak_sequence_start
    num_troughs_to_use = min(num_troughs, num_useable_peaks)
    trough_to_peak_metrics = pointToPointMetrics(
        start_point_indices=trough_indices[trough_sequence_start: trough_sequence_start + num_troughs_to_use],
        end_point_indices=peak_indices[peak_sequence_start: peak_sequence_start + num_troughs_to_use],
        point_values=value_data,
        point_times=time_stamps
    )
    trough_to_peak_metrics['p2p_order'] = 'trough_to_peak'

    # compute the peak to trough metrics
    peak_sequence_start = 0
    if first_peak_time < first_trough_time:
        trough_sequence_start = 0
    else:
        trough_sequence_start = 1
    num_peaks = len(peak_indices)
    num_useable_troughs = len(trough_indices) - trough_sequence_start
    num_peaks_to_use = min(num_peaks, num_useable_troughs)
    peak_to_trough_metrics = pointToPointMetrics(
        start_point_indices=peak_indices[peak_sequence_start: peak_sequence_start + num_peaks_to_use],
        end_point_indices=trough_indices[trough_sequence_start: trough_sequence_start + num_peaks_to_use],
        point_values=value_data,
        point_times=time_stamps
    )
    peak_to_trough_metrics['p2p_order'] = 'peak_to_trough'

    return (trough_to_peak_metrics, peak_to_trough_metrics)


def peakAndTroughIndices(
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

        # compute the width (in samples) from peak to peak or trough to trough that we expect the
        # signal to contain so we can eliminate noise components we presume will be shorter than this
        duration = time_stamps[-1] - time_stamps[0]
        num_samples = len(time_stamps)
        sampling_rate = float(num_samples)/duration
        expected_min_peak_width = sampling_rate/pacing_frquency_max_hz

        # compute the height from trough to peak or peak to trough that we expect the signal to contain.
        # we use this to eliminate noise components we pressume will be smaller than this.
        # note: it is probably not necessary to pass this parameter to the peak finder; as in,
        # it will likely work without this, and since it is much harder to estimate than the expected width,
        # and be a problem with for instance highly decaying signals and/or signals with significant noise,
        # it should be the first thing to consider changing (not using) if we're failing to pick all peaks/troughs.
        # also the use of HALF the calculated height of the middle-ish peak is entirely arbitrary and could
        # be replaced with some other fraction of a trough to peak height, or from a different place in the signal.
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


def extremePointIndices(signal: np.ndarray) -> Tuple[np.ndarray]:
    peaks, troughs = peakAndTroughIndices(signal)
    peaks_and_troughs = np.concatenate((peaks, troughs), axis=0)
    peaks_and_troughs_sorted = np.sort(peaks_and_troughs)
    return (peaks, troughs, peaks_and_troughs_sorted)


def ca2Data(path_to_data: str) -> Tuple[np.ndarray]:
    ''' Reads in an xlsx file containing ca2 experiment data and
        returns a tuple of numpy arrays (time stamps, signal) '''
    ca2_data = pd.read_excel(path_to_data, usecols=[1, 5], dtype=np.float32)
    ca2_data = ca2_data.to_numpy(copy=True).T
    return (ca2_data[0], ca2_data[1])


def lowPassFiltered(input_signal: np.ndarray, time_stamps: np.ndarray) -> np.ndarray:
    filter_order = 5  # how sharply the filter cuts off the larger the sharper it bends
    # frequency_range_hz = [1.0, 2.0]  # cut off frequency [start, stop] for bandpass/bandstop
    frequency_hz = 1.5
    duration = time_stamps[-1] - time_stamps[0]
    num_samples = len(time_stamps)
    sampleing_frequency = float(num_samples)/duration
    sos = butter(filter_order, frequency_hz, 'lowpass', fs=sampleing_frequency, output='sos')
    return sosfilt(sos, input_signal)
