# rf_svm_hybrid.py
# Hybrid model: Random Forest (feature selection) + SVM (classification)
# Dataset: lung_cancer_chemo.csv
# Target: "Response" (chemotherapy response)

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)
from sklearn.impute import SimpleImputer
import joblib


# ------------ CONFIG ------------

DATA_PATH = "lung_cancer_chemo.csv"   # change if your file has a different name
TARGET_COL = "Response"               # column with chemo response


# ------------ HELPER FUNCTIONS ------------

def load_data(path):
    print("=== Loading dataset ===")
    df = pd.read_csv(path)
    print(f"Shape (samples x features): {df.shape}")
    print("\nColumns:", list(df.columns))
    print("\nFirst 5 rows:\n", df.head())

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

    print("\nResponse label distribution:")
    print(df[TARGET_COL].value_counts())
    return df


def plot_feature_distributions(df, save_prefix="lung_chemo"):
    """Simple histograms for numeric features."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c != TARGET_COL]

    if len(numeric_cols) == 0:
        print("No numeric features to plot distributions.")
        return

    df[numeric_cols].hist(figsize=(14, 10))
    plt.suptitle("Feature Distributions", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_feature_histograms.png", dpi=300)
    plt.close()
    print(f"Saved feature distribution histograms -> {save_prefix}_feature_histograms.png")


def plot_correlation_heatmap(df, save_prefix="lung_chemo"):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c != TARGET_COL]

    if len(numeric_cols) < 2:
        print("Not enough numeric features for correlation heatmap.")
        return

    corr = df[numeric_cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_correlation_heatmap.png", dpi=300)
    plt.close()
    print(f"Saved correlation heatmap -> {save_prefix}_correlation_heatmap.png")


def prepare_data(df):
    # Separate features and target
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Encode target labels (PR, SD, etc.)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Check number of classes
    n_classes = len(np.unique(y_encoded))
    if n_classes < 2:
        print("\n[ERROR] The target variable has only ONE class.")
        print("You must use a dataset where 'Response' has at least TWO classes (e.g. PR vs SD).")
        sys.exit(1)

    print(f"\nNumber of classes in Response: {n_classes}")
    print("Encoded classes:", dict(zip(le.classes_, le.transform(le.classes_))))

    # We assume all non-target columns are numeric for this gene/clinical dataset.
    # If you have categorical features, you can one-hot encode them separately.
    numeric_cols = X.columns

    # Basic numeric imputation (median) just in case
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    return X_imputed, y_encoded, numeric_cols, le


def build_hybrid_model():
    """
    Hybrid model:
      - RandomForestClassifier used inside SelectFromModel for feature selection
      - SVM classifier on top of selected features
    """
    rf_for_selection = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    )

    svm_clf = SVC(
        kernel="rbf",
        probability=True,
        random_state=42,
        class_weight="balanced"
    )

    # Pipeline:
    # 1. Standardize features
    # 2. Use Random Forest to select important features
    # 3. Train SVM on selected features
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("feature_selection", SelectFromModel(rf_for_selection, threshold="median")),
        ("svm", svm_clf)
    ])

    return pipeline, rf_for_selection


def evaluate_model(model, X_train, X_test, y_train, y_test, label_encoder, save_prefix="lung_chemo"):
    # Train on train set
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    print("\n=== Classification Report (Test Set) ===")
    target_names = label_encoder.inverse_transform(sorted(np.unique(y_test)))
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=target_names)
    disp.plot()
    plt.title("Confusion Matrix - Hybrid RF + SVM")
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_confusion_matrix.png", dpi=300)
    plt.close()
    print(f"Saved confusion matrix -> {save_prefix}_confusion_matrix.png")

    # ROC curve (only for binary classification)
    if len(np.unique(y_test)) == 2:
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve - Hybrid RF + SVM")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_roc_curve.png", dpi=300)
        plt.close()
        print(f"Saved ROC curve -> {save_prefix}_roc_curve.png")
    else:
        print("Skipping ROC curve (requires binary classification).")


def cross_validate_model(model, X, y):
    print("\n=== Cross-validation (StratifiedKFold) ===")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print("Fold accuracies:", scores)
    print("Mean accuracy: {:.3f} ± {:.3f}".format(scores.mean(), scores.std()))


def plot_rf_feature_importance(rf_model, X, feature_names, save_prefix="lung_chemo"):
    """
    Fit a standalone Random Forest to show feature importances
    (for feature representation graph in your report).
    """
    rf_model.fit(X, y)
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(8, 6))
    plt.bar(range(len(feature_names)), importances[indices])
    plt.xticks(range(len(feature_names)), np.array(feature_names)[indices], rotation=45, ha="right")
    plt.title("Random Forest Feature Importances")
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_rf_feature_importances.png", dpi=300)
    plt.close()
    print(f"Saved Random Forest feature importance plot -> {save_prefix}_rf_feature_importances.png")


def main():
    # 1. Load and inspect data
    df = load_data(DATA_PATH)

    # 2. Basic visualizations of features
    plot_feature_distributions(df)
    plot_correlation_heatmap(df)

    # 3. Prepare data (X, y)
    X, y, feature_names, label_encoder = prepare_data(df)

    # 4. Train/test split (stratified so both sets have all classes)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    print("\nTrain size:", X_train.shape[0], " Test size:", X_test.shape[0])

    # 5. Build hybrid model
    hybrid_model, rf_for_importance = build_hybrid_model()

    # 6. Cross-validation on full data
    cross_validate_model(hybrid_model, X, y)

    # 7. Final train & evaluation on hold-out test set
    evaluate_model(hybrid_model, X_train, X_test, y_train, y_test,
                   label_encoder, save_prefix="lung_chemo_hybrid_rf_svm")

    # 8. Feature importance plot from RF
    plot_rf_feature_importance(rf_for_importance, X, feature_names,
                               save_prefix="lung_chemo_hybrid_rf_svm")

    # 9. Save model for later use
    joblib.dump(hybrid_model, "lung_chemo_hybrid_rf_svm_model.joblib")
    print("\nSaved trained model -> lung_chemo_hybrid_rf_svm_model.joblib")


if __name__ == "__main__":
    main()
