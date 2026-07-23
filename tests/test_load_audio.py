import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJ_ROOT))

from src.load_audio import load_audio
import pandas as pd
import pickle

data_dir = r"D:\Listening Effort Study\Audio Files"

print("-----STARTING INFORMAL TEST-----")

audio_dict = load_audio(audio_dir=data_dir, strict_naming=True)

print(f"Total audio files loaded: {len(audio_dict):,}")
print("Keys:", list(audio_dict.keys()))
print("Loaded keys/files:", list(audio_dict.keys())[:3], "... (showing first 3)")

# Peek at the first item in the dictionary to show structure
first_key = next(iter(audio_dict))
first_val = audio_dict[first_key]

print(f"\nSample key: {first_key}")

output = PROJ_ROOT/"data"/"processed_audio_signals.pkl"

output.parent.mkdir(parents=True, exist_ok=True)

with open(output, "wb") as f:
    pickle.dump(audio_dict, f)

print(f"Saved {len(audio_dict)} audio entries to {output}")

