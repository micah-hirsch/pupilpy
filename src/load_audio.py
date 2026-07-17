import glob
import os
import pandas as pd
import numpy as np
from scipy.io import wavfile

def load_audio(audio_dir, strict_naming = T):
    """
    Loads .wav files from a directory, extracts metadata, and returns a pandas Dataframe.
    """ 

    data_list = []
    signal_dict = {}

    wav_files = [f for f in os.listdir(audio_dir) if f.lower().endswith('.wav')]

    for file_name in wav_files:

        file_path = os.path.join(audio_dir, file_name)

        try:
            sample_rate, audio_data = wavfile.read(file_path)
            duration_ms = (len(audio_data)/sample_rate) * 1000
            name_noext, _ = os.path.splittext(file_name)
            parts = name_noext.split('_')

            speaker = None
            stimulus = None

            if strict_naming:

                if len(parts) >= 2:
                    id = parts[0]
                    stimulus = "_".join(parts[1:])

                else:
                    id = name_noext
                    print(f"Warning: '{file_name}' does not follow the 'ParticipantID_StimulusID.wav format. Assigned file_name to speaker")

            else:
                id = None

            signal_dict[file_name] = {
                "speaker": id,
                "stimulus_id": stimulus,
                "sample_rate": sample_rate,
                "audio_signal": audio_data,
                "duration_ms": duration_ms
            }
        
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    df = pd.DataFrame(data_list)

    return signal_dict
        



