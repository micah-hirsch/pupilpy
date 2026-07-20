import glob
import os
import pandas as pd
import numpy as np
from pathlib import Path


def parse_csv(file_path: str) -> pd.DataFrame:
    """
    Parses a single EyeLInk Data Viewer text/csv export file.
    Dynamically routes eye measurements
    """

    df = pd.read_csv(file_path, sep=None, engine='python', na_values=['.', 'MISSING', ''])

    df.columns = df.columns.astype(str).str.strip().str.lower().str.replace(r'[\s\-]+', '_', regex=True).str.replace(r'[^\w_]', '', regex=True)

    df = df.rename(columns={
        'recording_session_label': 'subject',
        'trial_index': 'trial'
    })

    df['pupil'] = np.nan
    df['blink'] = np.nan

    right_eye = df['eye_tracked'].str.lower() == 'right'
    left_eye = df['eye_tracked'].str.lower() == 'left'

    if 'right_pupil_size' in df.columns:
        df.loc[right_eye, 'pupil'] = df.loc[right_eye, 'right_pupil_size']
        df.loc[right_eye, 'blink'] = df.loc[right_eye, 'right_in_blink']
    
    if 'left_pupil_size' in df. columns:
        df.loc[left_eye, 'pupil'] = df.loc[left_eye, 'left_pupil_size']
        df.loc[left_eye, 'blink'] = df.loc[left_eye, 'left_in_blink']
    
    keep_columns = [
        'subject', 'trial', 'eye_tracked', 'blink', 'timestamp', 'pupil', 
        'ip_start_time', 'sample_message', 'effort_rating', 'code', 'speaker', 
        'practicetrial', 'targetphrase', 'counterbalance'
    ]

    return df[keep_columns]

def load_folder(folder_path: str = ".") -> pd.DataFrame:
    """
    Loops through a directory, processing each eye-tracking file, and merges them into a Dataframe
    """

    data_list = []
    directory = Path(folder_path)

    file_list = list(directory.glob("*.txt"))

    for file in file_list:
        
        df = parse_csv(file)

        data_list.append(df)
    
    if data_list:
        pupil_data = pd.concat(data_list, ignore_index = True)
        return pupil_data
    else:
        print("No .txt files found in the directory.")
        return pd.DataFrame()