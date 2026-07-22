import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

def pupil_smooth(pupil_signal, cutoff, fs, order = 1):
    """
    Applies a digital low-pass filter to a 1D array of data
    
    Parameters:
    -----------
    pupil_signal: pd.Series
        The 1D pupil dilation signal array.
    cutoff: int
        The cutoff frequency for the low-pass filter.
    fs: int
        The sampling rate of the pupil dilation recording.
    order: int, default 1
        The order of the filter.
    """

    temp = pupil_signal.interpolate(method="linear").ffill().bfill()

    if temp.isna().all():
        print(f"Data is missing for the entire trial.")
        return pupil_signal
    
    nyquist = fs/2
    wn = cutoff/nyquist
    
    coef_b, coef_a = butter(N=order, Wn=wn, btype = "low")

    filtered = filtfilt(coef_b, coef_a, temp, padlen = 300)

    filtered_final = np.where(pupil_signal.isna(), np.nan, filtered)

    return pd.Series(filtered_final, index=pupil_signal.index)


