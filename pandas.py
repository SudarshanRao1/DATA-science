import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("/mnt/data/TCGA_LAML_merged.csv")

# -----------------------------
# Step A: Convert survival columns to numeric
# -----------------------------
df['days_to_death'] = pd.to_numeric(df['days_to_death'], errors='coerce')
df['days_to_last_follow_up'] = pd.to_numeric(df['days_to_last_follow_up'], errors='coerce')

# -----------------------------
# Step B: Create overall survival time
# -----------------------------
df['overall_survival'] = df['days_to_death'].fillna(df['days_to_last_follow_up'])

# -----------------------------
# Step C: Create event label (VERY IMPORTANT)
# -----------------------------
# 1 = death occurred, 0 = censored (alive)
df['event'] = np.where(df['days_to_death'].notna(), 1, 0)

# Optional: remove rows with missing survival
df = df.dropna(subset=['overall_survival'])