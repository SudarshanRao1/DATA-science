import pandas as pd



# Load files

clinical = pd.read_csv("/content/clinical.tsv", sep="\t")

follow_up = pd.read_csv("/content/follow_up.tsv", sep="\t")

exposure = pd.read_csv("/content/exposure.tsv", sep="\t")

family = pd.read_csv("/content/family_history.tsv", sep="\t")



# -----------------------------

# 1. Standardize patient ID

# -----------------------------

clinical["patient_id"] = clinical["cases.submitter_id"]

follow_up["patient_id"] = follow_up["cases.submitter_id"]

exposure["patient_id"] = exposure["cases.submitter_id"]

family["patient_id"] = family["cases.submitter_id"]



# -----------------------------

# 2. Choose BEST follow-up column

# -----------------------------

if "follow_ups.days_to_progression" in follow_up.columns:

    time_col = "follow_ups.days_to_progression"

elif "follow_ups.days_to_progression_free" in follow_up.columns:

    time_col = "follow_ups.days_to_progression_free"

else:

    time_col = "follow_ups.days_to_follow_up"



print("Using follow-up column:", time_col)



# -----------------------------

# 3. One row per patient

# -----------------------------

follow_up_clean = (

    follow_up

    .sort_values(time_col)

    .drop_duplicates("patient_id", keep="last")

)



# -----------------------------
# 4. Merge all (with column cleanup)
# -----------------------------

# Define columns that appear in multiple files that we don't want to duplicate
redundant_cols = ["cases.submitter_id", "project.project_id", "cases.case_id"]

# Merge follow_up
merged = clinical.merge(
    follow_up_clean.drop(columns=[c for c in redundant_cols if c in follow_up_clean.columns]), 
    on="patient_id", 
    how="left"
)

# Merge exposure
merged = merged.merge(
    exposure.drop(columns=[c for c in redundant_cols if c in exposure.columns]), 
    on="patient_id", 
    how="left"
)

# Merge family
merged = merged.merge(
    family.drop(columns=[c for c in redundant_cols if c in family.columns]), 
    on="patient_id", 
    how="left"
)

print("Final merged shape:", merged.shape)


# -----------------------------

# 5. Save

# -----------------------------

merged.to_csv("/content/TCGA_LAML_merged.csv", index=False)

print("Saved TCGA_LAML_merged.csv")
