# train_gbm_lung_cancer.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
from sklearn.ensemble import GradientBoostingClassifier
import joblib
import os

# =============== 1. LOAD DATA & PROBLEM DEFINITION ==================
# Problem: Predict chemotherapy response in lung cancer patients
# Dataset: lung_cancer_chemo.csv
# Target: Response (0/1 or Responder/Non-Responder)

DATA_PATH = "lung_cancer_chemo.csv"  # <-- change if needed

df = pd.read_csv(DATA_PATH)

print("First 5 rows of dataset:")
print(df.head())
print("\nColumns in dataset:", df.columns.tolist())

# =============== 2. BASIC CLEANING & TARGET ENCODING ===============

TARGET_COL = "Response"  # <-- change if your target name is different

# Drop rows with missing target
df = df.dropna(subset=[TARGET_COL])

# If target is categorical, convert to 0/1
if df[TARGET_COL].dtype == "object":
    # Example: "Responder" -> 1, "Non-Responder" -> 0
    df[TARGET_COL] = df[TARGET_COL].map({
        "Responder": 1,
        "Non-Responder": 0
    }).fillna(df[TARGET_COL])  # if already numeric-like

y = df[TARGET_COL].astype(int)
X = df.drop(columns=[TARGET_COL])

# Identify numeric & categorical features
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

print("\nNumeric features:", numeric_features)
print("Categorical features:", categorical_features)

# =============== 3. FEATURE EXTRACTION / PREPROCESSING =============

numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# =============== 4. DEFINE GRADIENT BOOSTING MODEL =================

gbm_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    random_state=42
)

clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("gbm", gbm_model)
])

# =============== 5. TRAIN / TEST SPLIT ============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =============== 6. TRAINING ======================================

clf.fit(X_train, y_train)

# =============== 7. MODEL VALIDATION & METRICS =====================

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("\n=== Test Metrics ===")
print("Accuracy:", acc)
print("ROC-AUC:", roc_auc)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Cross-validation (for model validation section)
cv_scores = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
print("\n=== Cross-Validation ROC-AUC ===")
print("Scores:", cv_scores)
print("Mean ROC-AUC:", cv_scores.mean())
print("Std ROC-AUC:", cv_scores.std())

# =============== 8. PLOTS FOR REPORT ==============================

# ---- 8.1 Feature distribution (example for first 2 numeric features) ----
os.makedirs("plots", exist_ok=True)

for col in numeric_features[:2]:  # limit to first 2 numeric features
    plt.figure()
    df[col].hist(bins=20)
    plt.title(f"Feature Distribution: {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"plots/feature_distribution_{col}.png")
    plt.close()

# ---- 8.2 Confusion Matrix ----
cm = confusion_matrix(y_test, y_pred)

plt.figure()
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix")
plt.xlabel("Predicted label")
plt.ylabel("True label")
tick_marks = np.arange(2)
plt.xticks(tick_marks, ["Non-Responder", "Responder"])
plt.yticks(tick_marks, ["Non-Responder", "Responder"])

# Annotate cells
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j, i, format(cm[i, j], "d"),
            horizontalalignment="center",
            verticalalignment="center"
        )

plt.tight_layout()
plt.savefig("plots/confusion_matrix.png")
plt.close()

# ---- 8.3 ROC Curve ----
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

plt.figure()
plt.plot(fpr, tpr, label=f"GBM (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Gradient Boosting Model")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("plots/roc_curve.png")
plt.close()

# ---- 8.4 Feature Importance (after one-hot encoding) ----
# Get one-hot encoded feature names
onehot = clf.named_steps["preprocessor"].named_transformers_["cat"]["onehot"]
cat_feature_names = onehot.get_feature_names_out(categorical_features)
all_feature_names = numeric_features + list(cat_feature_names)

gbm = clf.named_steps["gbm"]
importances = gbm.feature_importances_

# Sort features by importance
indices = np.argsort(importances)[::-1]
top_n = 15  # show top 15 features
top_indices = indices[:top_n]

plt.figure(figsize=(8, 6))
plt.bar(range(top_n), importances[top_indices])
plt.xticks(range(top_n), [all_feature_names[i] for i in top_indices], rotation=90)
plt.title("Top Feature Importances - GBM")
plt.tight_layout()
plt.savefig("plots/feature_importance.png")
plt.close()

print("\nPlots saved in 'plots/' folder:")
print("- feature_distribution_<feature>.png")
print("- confusion_matrix.png")
print("- roc_curve.png")
print("- feature_importance.png")

# =============== 9. SAVE MODEL FOR DEPLOYMENT ======================

joblib.dump(clf, "lung_cancer_gbm_model.joblib")
print("\nModel saved as 'lung_cancer_gbm_model.joblib'")
