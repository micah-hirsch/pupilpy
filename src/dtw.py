import pandas as pd
from scipy.spatial.distance import euclidean
from scipy import signal
from fastdtw import fastdtw

def audio_spectrogram(audio_data, audio_sr, target_sr=1000):
    """
    Converts a raw waveform into a spectogram matrix to downsample the audio
    to ta specific sampling rate.

    Parameters:
    -----------
    audio_data : np.ndarray
        The raw 1D or 2D audio signal array
    audio_st: int
        The sampling rate of the raw audio
    target_sr : int, default 1000
        The desired sampling rate for the output timeline (e.g., 1000 for
        pupillometry sync, 500 for standard eye-trackers).
    
    Returns:
    --------
    np.ndarray
        A 2D spectrogram matrix of shape (num_target_frames, num_frequency_bins).
    """

    # Converting multi-channel recordings into mono
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis = 1)

    # Calculating sample hop size
    hop_size = int(audio_sr/target_sr)

    if hop_size < 1:
        raise ValueError(f"Target sample rate ({target_sr} Hz) cannot be higher than the initial sample rate ({audio_sr} Hz).")

    window_size = hop_size*4
    overlap = window_size - hop_size

    # Compute STFT
    frequencies, times, stft_matrix = signal.stft(
        audio_data,
        fs = audio_sr,
        nperseg = window_size,
        noverlap = overlap
    )

    features = np.abs(stft_matrix).T

    return features

def align_audio(template_audio, comp_audio, audio_sr, target_sr = 1000):
    """
    Downsamples two raw audio arrays to a matching target sample rate 
    spectrogram matrix, and calculates the optimal DTW warping path.

    Parameters:
    -----------
    template_audio : np.ndarray
        Raw audio array for the template audio file
    comp_audio : np.ndarray
        Raw audio array for the comparison speaker
    audio_sr : int
        The native sampling rate of the wav files
    target_sr : int, default 1000
        The frame rate matching your eye-tracking sample rate (e.g. 1000 Hz)

    Returns:
    --------
    dtw_distance : float
        The absolute cumulative warping cost between the signals
    warp_path : list of tuples
        The index aligment pairs mapping the target_sr timelines.
    """

    # Transforming the audio files into the downsampled spectrograms
    spectro_a = audio_spectrogram(template_audio, audio_sr, target_sr)
    spectro_b = audio_spectrogram(comp_audio, audio_sr, target_sr)

    # Running fastDTW on the 2D features of the spectrogram
    dtw_distance, warp_path = fastdtw(spectro_a, spectro_b, dist=euclidean)

    return dtw_distance, warp_path

def run_dtw(signal_dict, template_speaker, target_sr=1000):
    """
    Loops through the signal_dict to automatically find matching phrase pairs, runs the downsampled
    spectrogram dynamic time warping against the template speaker recording, and appends the dictionary with 
    the warp paths and distances.

    Parameters:
    -----------
    signals_dict : dict
        The multi-dimensional dictionary containing the .wav audio file data
    template_speaker : str
        The ID or code of the speaker to use as the alignment baseline for phrase pairs (e.g., "Control")
    target_sr : int, default 1000
        The target sample rate matching the pupillometry data 
    """

    template_phrases = {}
    for filename, meta in signal_dict.items():
        if meta["speaker"] == template_speaker:
            template_phrases[meta["stimulus_id"]] = meta
            meta["warp_path"] = None
            meta["dtw_distance"] = 0.0
    
    print(f"Found {len(template_phrases)} baseline phrase templates for '{template_speaker}'.")
    print(f"Automating DTW alignments for remaining comparison files...")

    for file_name, meta in signal_dict.items():
        if meta["participant_id"] == template_speaker:
            continue

        stim_id = meta["stimulus_id"]

        if stim_id in template_phrases:
            temp_meta = template_phrases[stim_id]
            initial_sr = temp_meta["sample_rate"]

            try:
                distance, warp_path = align_audio(
                    template_audio = temp_meta["audio_signal"],
                    comp_audio = meta["audio_signal"],
                    audio_sr = initial_sr,
                    target_sr = target_sr
                )

                meta["warp_path"] = warp_path
                meta["dtw_distance"] = distance
                meta["duration_diff_ms"] = abs(temp_meta["duration_ms"] - meta["duration_ms"])
            
            except Exception as e:
                print(f"Error executing DTW for {file_name}: {e}")
                meta["warp_path"] = None
                meta["dtw_distance"] = None
        else:
            print(f"Skipping {file_name}: No template phrase match found for '{stim_id}'")
            meta["warp_path"] = None
            meta["dtw_distance"] = None
    
    print("Success! Your signals_dict has been updated with warp paths.")