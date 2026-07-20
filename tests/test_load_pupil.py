import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJ_ROOT))

from src.load_pupil import parse_csv, load_folder
import pandas as pd

data_dir = r"D:\Listening Effort Study\Raw Data\Extracted Pupil and PLE Ratings"

print("-----STARTING INFORMAL TEST-----")

txt_files = list(Path(data_dir).glob("*.txt"))

if txt_files:
    first_file = str(txt_files[0])
    print(f"\n[1] Testing parse_csv() on: {Path(first_file).name}")

    single_df = parse_csv(first_file)
    print("Single file loaded successfully!")
    print(f"Shape: {single_df.shape}")
    print("Columns:", list(single_df.columns))
    print("\nFirst 3 rows:")
    print(single_df.head(3))
else:
    print(f"No .txt file found in {data_dir}")

print("----TESTING LOAD FOLDER-----")

full_df = load_folder(data_dir)

print("\n--- SUMMARY OF COMBINED DATAFRAME ---")
print(f"Total rows: {len(full_df):,}")
print(f"Total columns: {len(full_df.columns)}")
print("Columns present:", list(full_df.columns))

if not full_df.empty and 'subject' in full_df.columns:
    print(f"\nUnique Subjects Loaded ({full_df['subject'].nunique()} total):")
    print(full_df['subject'].unique())