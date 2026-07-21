import pandas as pd
import numpy as np
from typing import Literal

def align_time(pupil_signal, signal_dict, position: Literal["onset", "offset"], sample_message, baseline = 3000, retention = 3000):
    """
    Align the pupil records so that time = 0 represents either the onset or the offset of the stimuli
    """
    
    audio_info = pd.DataFrame({
        "speaker": [signal_dict["speaker"]],
        "code": [signal_dict["stimulus_id"]],
        "duration": [signal_dict["duration_ms"]]
    })

    landmarks = pd.merge(pupil_signal, audio_info, on = ['speaker', 'code'], how = 'outer')
    landmarks = landmarks[landmarks['sample_message'].str.contains(sample_message, na=False)]

    if position == "onset":
        landmarks["phrase_start"] = landmarks["timestamp"]
        landmarks["trial_start"] = (landmarks["phrase_start"] - baseline).round()
        landmarks["phrase_end"] = (landmarks["phrase_start"] + landmarks["duration"]).round()
        landmarks["trial_end"] = (landmarks["phrase_end"] + retention).round()

        landmarks = landmarks[["subject", "trial", "phrase_start", "trial_start", "phrase_end", "trial_end"]]

        data = pd.merge(pupil_signal, landmarks, on = ['subject', 'trial'], how="left")

        trimmed_data = data[(data["timestamp"] >= data["trial_start"]) & (data["timestamp"] <= data["trial_end"])].copy()
        trimmed_data["time"] = trimmed_data["timestamp"] - trimmed_data["phrase_start"]

    if position == "offset":
        landmarks["phrase_end"] = landmarks["timestamp"]
        landmarks["phrase_start"] = (landmarks["phrase_end"] - landmarks["duration"]).round()
        landmarks["trial_start"] = (landmarks["phrase_start"] - baseline).round()
        landmarks["trial_end"] = (landmarks["phrase_end"] + retention).round()

        landmarks = landmarks[["subject", "trial", "phrase_start", "trial_start", "phrase_end", "trial_end"]]

        data = pd.merge(pupil_signal, landmarks, on = ['subject', 'trial'], how="left")

        trimmed_data = data[(data["timestamp"] >= data["trial_start"]) & (data["timestamp"] <= data["trial_end"])].copy()
        trimmed_data["time"] = trimmed_data["timestamp"] - trimmed_data["phrase_end"]
    
    trimmed_data = trimmed_data.drop(columns=["timestamp", "phrase_start", "trial_start", "phrase_end", "trial_end"])

    return trimmed_data


