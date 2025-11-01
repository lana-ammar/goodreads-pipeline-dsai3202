import os
import pandas as pd
import glob

print("\n=== TRIAL SCRIPT STARTED ===")

# --- Environment setup ---
INPUT_DIR = os.environ.get("INPUT_DIR", "/opt/ml/processing/input/features")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/opt/ml/processing/output")
ROW_CHUNK = int(os.environ.get("ROW_CHUNK", "2000"))

print(f"INPUT_DIR: {INPUT_DIR}")
print(f"OUTPUT_DIR: {OUTPUT_DIR}")
print(f"ROW_CHUNK: {ROW_CHUNK}")

# --- Recursively search for .parquet or .parquet.snappy files ---
parquet_files = glob.glob(os.path.join(INPUT_DIR, "**", "*.parquet*"), recursive=True)
print(f"\nFound {len(parquet_files)} parquet files.")
if parquet_files:
    print("Sample files:")
    for f in parquet_files[:5]:
        print("  •", f)
else:
    raise FileNotFoundError("No .parquet files found in input (even after recursive search)!")

# --- Load a few parquet files until we reach ROW_CHUNK rows ---
dfs = []
total_rows = 0

for file in parquet_files:
    try:
        df_part = pd.read_parquet(file)
        dfs.append(df_part)
        total_rows += len(df_part)
        print(f"Loaded {len(df_part)} rows from {os.path.basename(file)} (total so far: {total_rows})")
    except Exception as e:
        print(f"⚠️ Skipping {file} due to error: {e}")
    if total_rows >= ROW_CHUNK:
        break

if not dfs:
    raise ValueError("No readable parquet files were loaded!")

# --- Combine and trim to desired size ---
df = pd.concat(dfs, ignore_index=True)
df_trial = df.head(ROW_CHUNK)

# --- Print schema info for validation ---
print("\n=== DATA SUMMARY ===")
print(f"Total rows loaded: {len(df)}")
print(f"Trial sample rows: {len(df_trial)}")
print(f"Number of columns: {df_trial.shape[1]}")
print("\nColumn names:")
print(list(df_trial.columns))

# --- Save output to /opt/ml/processing/output ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_path = os.path.join(OUTPUT_DIR, "trial_output.parquet")
df_trial.to_parquet(output_path, index=False)

print(f"\nSaved trial dataset ({len(df_trial)} rows) to: {output_path}")
print("\n=== TRIAL SCRIPT COMPLETED SUCCESSFULLY ===")
