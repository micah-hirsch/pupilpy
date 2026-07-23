import numpy as np
import pandas as pd
from typing import Literal

def inter_pupil(pupil_signal, 
                fs=1000, 
                method: Literal["linear", "cubic", "spline", "nearest"] = "linear", 
                max_gap = 1000.0, 
                order = 2):
    """
    Interpolate missing pupil data up to a maximum gap duration defined by the user in milliseconds.
    """

    limit = int(np.round((max_gap)/1000.0 * fs))

    interp_kwargs = {
        "method": method,
        "limit": limit,
        "limit_area": "inside"
    }

    if method == "spline":
        interp_kwargs["order"] = order

    interpolated_pupil = pupil_signal.interpolate(**interp_kwargs)

    return pd.Series(interpolated_pupil, index=pupil_signal.index)
