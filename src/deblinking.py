import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation

def pupil_detect(pupil_signal, time_signal, threshold=8.0):
    """
    Detects blink artifacts using Median absolute deviation velocity thresholds and returns a 
    Boolean series (True for blink/artifact, False for valid points).
    """
    # Calculating time step and velocity
    dt = time_signal.diff()
    dilation_vel = pupil_signal.diff()/dt

    # Calculating MAD of velocity
    vel_median = dilation_vel.median(skipna=True)
    v_mad = (dilation_vel - vel_median).abs().median(skipna=True)

    # Isolating negative velocity
    dil_neg = np.where(dilation_vel >= 0, 0.0, dilation_vel)

    # Mask missing samples OR samples exceeding threshold
    vel_artifact = np.abs(dil_neg) > (threshold * v_mad)
    is_blink = pupil_signal.isna() | vel_artifact

    return pd.Series(is_blink, index=pupil_signal.index)


def pupil_extend(pupil_signal, is_blink_mask, fs=1000, before_ms=160, after_ms=50):
    """
    Extends blink boundaries backward and forward in time based on a millisecond window.
    Returns pupil_signal with extended NA regions.
    """
    # Convert ms to sample counts based on sampling rate fs
    samples_before = int(np.round((before_ms/1000.0)*fs))
    samples_after = int(np.round((after_ms/1000.0)*fs))

    footprint = np.concatenate([
        np.ones(samples_before, dtype=bool),
        [True],
        np.ones(samples_after, dtype=bool)
    ])

    extend_mask = binary_dilation(is_blink_mask, structure=footprint)
    extended_signal = np.where(extend_mask, np.nan, pupil_signal)

    return pd.Series(extended_signal, index = pupil_signal.index)
