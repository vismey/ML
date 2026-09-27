"""
train.py
Random Forest Ensemble Training & Evaluation Pipeline:
Exclusively powered by Random Forest Classifier (Learning with Trees: Ensemble Bagging).
Trained on AI_Human.csv (Balanced 10,000 samples).
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless backend for server/script rendering
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

import features

PLOTS_DIR = os.path.join("static", "plots")
MODELS_DIR = "models"
DATA_PATH = os.path.join("data", "ai_vs_human_dataset.csv")
AI_HUMAN_PATH = "AI_Human.csv"
SAMPLED_10K_PATH = os.path.join("data", "ai_human_10k_sampled.csv")
FEATURES_CACHE_PATH = os.path.join("data", "features_10k_cache.joblib")


def setup_directories():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs("data", exist_ok=True)


def train_and_evaluate():
    setup_directories()
    print("=" * 70)
    print("AI vs Human Text Detector: Random Forest Model Training & Evaluation")
    print("=" * 70)

    # 1. Load Dataset (Prioritizing balanced sample from AI_Human.csv)
    if os.path.exists(SAMPLED_10K_PATH):
        print(f"Loading balanced 10,000-sample dataset from {SAMPLED_10K_PATH}...")
        df = pd.read_csv(SAMPLED_10K_PATH)
    elif os.path.exists(AI_HUMAN_PATH):
        print("Found AI_Human.csv. Preparing balanced 10,000-sample dataset (5,000 Human, 5,000 AI)...")
        df_raw = pd.read_csv(AI_HUMAN_PATH)
        df_raw = df_raw.dropna(subset=["text", "generated"])
        human_df = df_raw[df_raw["generated"] == 0].sample(n=min(5000, int((df_raw["generated"] == 0).sum())), random_state=42)
        ai_df = df_raw[df_raw["generated"] == 1].sample(n=min(5000, int((df_raw["generated"] == 1).sum())), random_state=42)
        df = pd.concat([human_df, ai_df]).sample(frac=1, random_state=42).reset_index(drop=True)
        df["label"] = df["generated"].astype(int)
        df = df[["text", "label"]]
        df.to_csv(SAMPLED_10K_PATH, index=False)
        print(f"Saved balanced 10,000-sample dataset to {SAMPLED_10K_PATH}.")
    elif os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        import dataset_generator
        df = dataset_generator.expand_dataset()
        df.to_csv(DATA_PATH, index=False)

    print(f"Loaded dataset: {len(df)} samples ({sum(df['label'] == 0)} Human, {sum(df['label'] == 1)} AI)")

    # 2. Extract Handcrafted Statistical Features (Multi-Core Parallel)
    if os.path.exists(FEATURES_CACHE_PATH):
        print(f"Loading cached features from {FEATURES_CACHE_PATH}...")
        X_features_df = joblib.load(FEATURES_CACHE_PATH)
    else:
        print("Extracting handcrafted statistical & linguistic features (multi-core parallel)...")
        X_features_df = features.extract_features_dataframe(df["text"])
        joblib.dump(X_features_df, FEATURES_CACHE_PATH)
        print(f"Saved extracted features cache to {FEATURES_CACHE_PATH}.")

    y = df["label"].values
    feature_names = features.FEATURE_NAMES

    # 3. Train/Test Split (80% Train, 20% Test) with Stratification
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_features_df.values, y, test_size=0.20, random_state=42, stratify=y
    )

    # Standard Scaler (Zero mean, unit variance)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # =========================================================================
    # DIMENSIONALITY REDUCTION (PCA 2D PROJECTION)
    # =========================================================================
    print("\n--- Dimensionality Reduction (PCA) ---")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_train)
    explained_variance = pca.explained_variance_ratio_
    print(f"PCA 2D Explained Variance: PC1 = {explained_variance[0]:.2%}, PC2 = {explained_variance[1]:.2%} (Total = {sum(explained_variance):.2%})")

    # Plot PCA 2D Scatter
    plt.figure(figsize=(8, 6))
    plot_indices = np.random.RandomState(42).choice(len(X_pca), size=min(1000, len(X_pca)), replace=False)
    plt.scatter(X_pca[plot_indices][y_train[plot_indices] == 0, 0], X_pca[plot_indices][y_train[plot_indices] == 0, 1], c='#10b981', label='Human Texts (Class 0)', alpha=0.6, s=30)
    plt.scatter(X_pca[plot_indices][y_train[plot_indices] == 1, 0], X_pca[plot_indices][y_train[plot_indices] == 1, 1], c='#6366f1', label='AI Texts (Class 1)', alpha=0.6, s=30)
    plt.title(f"PCA 2D Feature Space Projection\nExplained Variance: {sum(explained_variance):.1%}", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel(f"Principal Component 1 ({explained_variance[0]:.1%} var)", fontsize=11)
    plt.ylabel(f"Principal Component 2 ({explained_variance[1]:.1%} var)", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(frameon=True, facecolor='white', loc='best')
    plt.tight_layout()
    pca_plot_path = os.path.join(PLOTS_DIR, "pca_projection.png")
    plt.savefig(pca_plot_path, dpi=200)
    plt.close()

    # =========================================================================
    # RANDOM FOREST ENSEMBLE TRAINING & 5-FOLD CV
    # =========================================================================
    print("\n--- Random Forest Ensemble Training ---")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )

    # 5-Fold Stratified Cross-Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
    cv_mean = float(cv_scores.mean())
    cv_std = float(cv_scores.std())

    # Fit Random Forest on Training Set
    rf_model.fit(X_train, y_train)

    # Predictions & Performance on Test Set
    y_pred = rf_model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred).tolist()
    oob = float(rf_model.oob_score_)

    print(f"Test Accuracy:  {acc:.2%}")
    print(f"Precision:      {prec:.4f}")
    print(f"Recall:         {rec:.4f}")
    print(f"F1-Score:       {f1:.4f}")
    print(f"5-Fold CV F1:   {cv_mean:.4f} (+/- {cv_std:.4f})")
    print(f"OOB Score:      {oob:.4f}")

    # =========================================================================
    # PLOT 1: CONFUSION MATRIX HEATMAP (RANDOM FOREST)
    # =========================================================================
    plt.figure(figsize=(6, 5))
    cm_arr = np.array(cm)
    plt.imshow(cm_arr, cmap='Blues', interpolation='nearest')
    plt.title(f"Random Forest Confusion Matrix\n(Test Set N={len(y_test)})", fontsize=13, fontweight='bold', pad=12)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xticks([0, 1], ['Human', 'AI'], fontsize=10)
    plt.yticks([0, 1], ['Human', 'AI'], fontsize=10)
    plt.xlabel("Predicted Class", fontsize=11, labelpad=8)
    plt.ylabel("Actual True Class", fontsize=11, labelpad=8)

    for i in range(2):
        for j in range(2):
            cell_color = "white" if cm_arr[i, j] > cm_arr.max() / 2 else "#1e293b"
            plt.text(j, i, f"{cm_arr[i, j]}", ha="center", va="center",
                     color=cell_color, fontsize=13, fontweight='bold')

    plt.tight_layout()
    cm_plot_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=200)
    plt.close()

    # =========================================================================
    # PLOT 2: HANDCRAFTED FEATURE IMPORTANCE (GINI IMPORTANCE)
    # =========================================================================
    importances = rf_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    clean_labels = [features.FEATURE_LABELS[feature_names[i]] for i in sorted_idx]
    y_pos = np.arange(len(clean_labels))
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(clean_labels)))[::-1]
    plt.barh(y_pos, importances[sorted_idx], color=colors, edgecolor='none')
    plt.yticks(y_pos, clean_labels, fontsize=9)
    plt.gca().invert_yaxis()
    plt.title("Random Forest Handcrafted Feature Importance (Gini Importance)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Relative Importance Score", fontsize=11)
    plt.grid(True, axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    fi_plot_path = os.path.join(PLOTS_DIR, "feature_importance.png")
    plt.savefig(fi_plot_path, dpi=200)
    plt.close()

    # =========================================================================
    # PLOT 3: ENSEMBLE VARIANCE REDUCTION (TREES vs OOB & TEST ERROR)
    # =========================================================================
    n_trees_range = [10, 25, 50, 75, 100, 150]
    oob_errors = []
    test_errors = []

    for n in n_trees_range:
        temp_rf = RandomForestClassifier(n_estimators=n, max_depth=8, oob_score=True, random_state=42, n_jobs=-1)
        temp_rf.fit(X_train, y_train)
        oob_errors.append(1.0 - temp_rf.oob_score_)
        test_errors.append(1.0 - accuracy_score(y_test, temp_rf.predict(X_test)))

    plt.figure(figsize=(8, 5))
    plt.plot(n_trees_range, oob_errors, marker='o', color='#ea580c', linewidth=2, label='Out-Of-Bag (OOB) Error')
    plt.plot(n_trees_range, test_errors, marker='s', color='#2563eb', linewidth=2, linestyle='--', label='Test Set Error')
    plt.title("Random Forest Ensemble: Number of Trees vs Error Rate\n(Demonstrating Variance Reduction via Bagging)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Number of Estimators / Decision Trees (n_estimators)", fontsize=11)
    plt.ylabel("Classification Error Rate", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(frameon=True, facecolor='white')
    plt.tight_layout()
    trees_plot_path = os.path.join(PLOTS_DIR, "rf_trees_analysis.png")
    plt.savefig(trees_plot_path, dpi=200)
    plt.close()

    # =========================================================================
    # PERSIST MODEL ARTIFACTS
    # =========================================================================
    joblib.dump(rf_model, os.path.join(MODELS_DIR, "random_forest_model.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "feature_scaler.joblib"))
    joblib.dump(pca, os.path.join(MODELS_DIR, "pca_transformer.joblib"))

    # Remove extraneous model files if present
    for extra_file in ["logistic_regression.joblib", "decision_tree.joblib"]:
        p = os.path.join(MODELS_DIR, extra_file)
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass

    # Save metrics JSON for frontend display
    metrics_data = {
        "model_name": "Random Forest Ensemble Classifier",
        "dataset_name": "AI_Human.csv (Balanced 10k Sample)",
        "dataset_size": len(df),
        "test_samples": len(y_test),
        "features_count": len(feature_names),
        "feature_names": feature_names,
        "feature_labels": features.FEATURE_LABELS,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "cv_f1_mean": round(cv_mean, 4),
        "cv_f1_std": round(cv_std, 4),
        "oob_score": round(oob, 4),
        "confusion_matrix": cm,
        "pca_variance": [round(float(v), 4) for v in explained_variance]
    }

    with open(os.path.join(MODELS_DIR, "metrics_report.json"), "w") as f:
        json.dump(metrics_data, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Random Forest Training Complete! Test Accuracy: {acc:.2%}, F1-Score: {f1:.4f}")
    print(f"Saved artifacts to: {MODELS_DIR}/")
    print(f"Saved plots to: {PLOTS_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    train_and_evaluate()
