#!/usr/bin/env python3
"""
Train an L1-regularized model (Lasso / L1 Logistic Regression)
for lung cancer chemotherapy response using the SAME dataset as your GBM model.

Usage:
    python3 lasoo.py

Requirements:
    pip install pandas numpy scikit-learn matplotlib joblib
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    KFold
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    mean_squared_error,
    r2_score
)
import joblib

# ==========================
# 1. CONFIG: CHANGE THESE IF NEEDED
# ==========================
# Use the SAME CSV and target column as your GBM model
DATA_PATH = "lung_cancer_chemo.csv"   # <--- change to your actual file
TARGET_COL = "Response"                    # <--- change if target column name differs

TEST_SIZE = 0.2
RANDOM_STATE = 42
OUTPUT_DIR = "outputs_lasso"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data(path, target_col):
    """Load dataset and split into features + target."""
    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return df, X, y


def detect_feature_types(X):
    """Return numeric and categorical column name lists."""
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["int64", "float64"]).columns.tolist()
    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols, categorical_cols):
    """Build preprocessing transformer (impute + scale / one-hot)."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ],
        remainder="drop"
    )
    return preprocessor


def is_classification_task(y):
    """Decide if we do classification or regression."""
    # If y is non-numeric or has few unique values, treat as classification
    if y.dtype == "object":
        return True
    # If numeric but has very few unique values, also treat as classification
    if y.nunique() <= 10 and set(y.unique()).issubset({0, 1, 2, 3, 4, 5}):
        return True
    return False


def get_feature_names_from_preprocessor(preprocessor, numeric_cols, categorical_cols):
    """Recover feature names after ColumnTransformer + OneHotEncoder."""
    feature_names = []

    # numeric part
    if "num" in preprocessor.named_transformers_:
        feature_names.extend(numeric_cols)

    # categorical part
    if "cat" in preprocessor.named_transformers_:
        cat_pipeline = preprocessor.named_transformers_["cat"]
        ohe = cat_pipeline.named_steps["onehot"]
        ohe_feature_names = ohe.get_feature_names_out(categorical_cols)
        feature_names.extend(list(ohe_feature_names))

    return feature_names


def plot_top_l1_coefficients(model, feature_names, output_path, top_n=20):
    """Plot top N absolute coefficient magnitudes for L1 model."""
    if hasattr(model, "coef_"):
        # Classification (LogisticRegression)
        coef = model.coef_
        # If multiclass, take norm across classes
        if coef.ndim > 1:
            coef = np.linalg.norm(coef, axis=0)
        else:
            coef = coef.flatten()
    elif hasattr(model, "feature_importances_"):
        coef = model.feature_importances_
    else:
        # Regression (Lasso)
        coef = model.coef_

    coef = np.array(coef)
    abs_coef = np.abs(coef)

    idx = np.argsort(abs_coef)[::-1][:top_n]
    top_features = np.array(feature_names)[idx]
    top_values = coef[idx]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top_features)), top_values)
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel("Coefficient value")
    plt.title(f"Top {top_n} L1 coefficients")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_confusion_matrix_and_roc(y_test, y_pred, y_proba, output_prefix):
    """For classification: confusion matrix + ROC curve."""
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(cm.shape[0])
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="white")
    plt.tight_layout()
    plt.savefig(output_prefix + "_confusion_matrix.png")
    plt.close()

    # ROC curve (binary only)
    if y_proba is not None and len(np.unique(y_test)) == 2:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.figure(figsize=(5, 4))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (L1 model)")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(output_prefix + "_roc_curve.png")
        plt.close()


def plot_regression_results(y_test, y_pred, output_prefix):
    """For regression: actual vs predicted scatter + residuals."""
    # Actual vs predicted
    plt.figure(figsize=(5, 4))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    plt.title("Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(output_prefix + "_actual_vs_pred.png")
    plt.close()

    # Residual plot
    residuals = y_test - y_pred
    plt.figure(figsize=(5, 4))
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted")
    plt.tight_layout()
    plt.savefig(output_prefix + "_residuals.png")
    plt.close()


def main():
    print("=== L1 model for Lung Cancer Chemotherapy Response ===")
    print(f"Loading data from: {DATA_PATH}")

    df, X, y = load_data(DATA_PATH, TARGET_COL)
    print("Dataset shape:", df.shape)

    numeric_cols, categorical_cols = detect_feature_types(X)
    print("Numeric columns:", numeric_cols)
    print("Categorical columns:", categorical_cols)

    # Decide classification or regression
    classification = is_classification_task(y)
    print("Task type:", "Classification" if classification else "Regression")

    label_encoder = None
    if classification:
        if y.dtype != "int64" and y.dtype != "float64":
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            print("Label classes:", list(label_encoder.classes_))

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    # ==========================
    # 2. Define L1 model + pipeline
    # ==========================
    if classification:
        l1_model = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            max_iter=500,
            random_state=RANDOM_STATE
        )
        param_grid = {
            "model__C": [0.01, 0.1, 1.0, 10.0]
        }
    else:
        l1_model = Lasso(random_state=RANDOM_STATE, max_iter=5000)
        param_grid = {
            "model__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0]
        }

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", l1_model)
    ])

    # ==========================
    # 3. Train-test split
    # ==========================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y if classification else None
    )

    # ==========================
    # 4. GridSearchCV
    # ==========================
    print("Running GridSearchCV...")
    if classification:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        scoring = "f1_macro"
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        scoring = "neg_mean_squared_error"

    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)

    print("Best parameters:", grid.best_params_)
    print("Best CV score:", grid.best_score_)

    best_model = grid.best_estimator_

    # ==========================
    # 5. Evaluation on test set
    # ==========================
    y_pred = best_model.predict(X_test)

    if classification:
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        print(f"Test Accuracy: {acc:.4f}")
        print(f"Test F1-macro: {f1:.4f}")

        # Probabilities for ROC
        try:
            y_proba = best_model.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) == 2 else None
        except Exception:
            y_proba = None

        plot_confusion_matrix_and_roc(
            y_test,
            y_pred,
            y_proba,
            os.path.join(OUTPUT_DIR, "lasso_classification")
        )
    else:
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        print(f"Test RMSE: {rmse:.4f}")
        print(f"Test R2:   {r2:.4f}")

        plot_regression_results(
            y_test,
            y_pred,
            os.path.join(OUTPUT_DIR, "lasso_regression")
        )

    # ==========================
    # 6. Feature importance plot
    # ==========================
    # Extract fitted preprocessor to get feature names
    fitted_preprocessor = best_model.named_steps["preprocessor"]
    feature_names = get_feature_names_from_preprocessor(
        fitted_preprocessor, numeric_cols, categorical_cols
    )
    core_model = best_model.named_steps["model"]

    plot_top_l1_coefficients(
        core_model,
        feature_names,
        os.path.join(OUTPUT_DIR, "lasso_top_coeffs.png"),
        top_n=20
    )

    # ==========================
    # 7. Save model + metadata
    # ==========================
    save_path = os.path.join(OUTPUT_DIR, "lasso_model.joblib")
    joblib.dump(
        {
            "pipeline": best_model,
            "label_encoder": label_encoder,
            "feature_names": feature_names,
            "classification": classification,
            "best_params": grid.best_params_,
            "cv_best_score": grid.best_score_
        },
        save_path
    )
    print(f"Model and metadata saved to: {save_path}")
    print(f"Plots saved in folder: {OUTPUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
