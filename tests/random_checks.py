import pandas as pd
import numpy as np

# Load all bark segments from CSV
df_bark = pd.read_csv("data/processed/bark_segments.csv")

# Load embeddings that were actually extracted
X = np.load("data/processed/X.npy")

# Get YouTube IDs that were successfully extracted
ytid_extracted = set(df_bark.iloc[:len(X)]["YTID"])  # Only the first 24

# Get all YouTube IDs that should have been extracted
ytid_all = set(df_bark["YTID"])

# Find missing YouTube IDs
missing_ytids = ytid_all - ytid_extracted

print(f"Total expected: {len(ytid_all)}")
print(f"Total extracted: {len(ytid_extracted)}")
print(f"Missing: {len(missing_ytids)}")

# Print the missing segments
print("\nMissing segments:")
print(df_bark[df_bark["YTID"].isin(missing_ytids)])
