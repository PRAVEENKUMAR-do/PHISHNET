#!/usr/bin/env python
"""
PhishNet train.py - CLEANED VERSION
Dataset: reduced_dataset.csv (url, label)
Target Model: RandomForestClassifier
"""
import pandas as pd
import numpy as np
import joblib, os, sys, warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")

# Ensure project modules are available
sys.path.insert(0, ".")
from src.features import extract_features, FEATURE_NAMES

def main():
    print("=" * 60)
    print("  PhishNet - Fixed Training Pipeline")
    print("=" * 60)

    # ── TASK 1 & 2: Load New Dataset ──────────────────────────
    dataset_path = "data/reduced_dataset.csv"
    if not os.path.exists(dataset_path):
        print(f"  ERROR: {dataset_path} not found!")
        return

    print(f"\n[1/5] Loading new dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"  Rows: {len(df):,}  Columns: {list(df.columns)}")

    # ── TASK 3: Fix Label Handling ────────────────────────────
    # Labels: 0 (SAFE), 1 (MALICIOUS)
    print("\n[2/5] Preparing data labels...")
    if 'label' not in df.columns:
        print("  ERROR: 'label' column missing in dataset!")
        return
    
    # Ensure labels are binary integers
    df['label'] = df['label'].astype(int)
    n0 = (df['label'] == 0).sum()
    n1 = (df['label'] == 1).sum()
    print(f"  SAFE (0):      {n0:>8,}")
    print(f"  MALICIOUS (1): {n1:>8,}")

    # ── TASK 4: Feature Extraction (Optimized) ──────────────────
    print("\n[3/5] Extracting features (no cache)...")
    rows = []
    total = len(df)
    
    # Pre-optimize loop: List-based iteration is much faster than iterrows
    urls = df['url'].fillna('').astype(str).tolist()
    
    for i, url in enumerate(urls):
        # Handle empty/invalid URLs before extraction
        if not url.strip():
            f = [0.0] * len(FEATURE_NAMES)
        else:
            try:
                f = extract_features(url)
            except Exception:
                f = [0.0] * len(FEATURE_NAMES)
        rows.append(f)
        
        if (i + 1) % 5000 == 0:
            print(f"    Processed {i + 1:>6,}/{total:,} URLs...")

    X = np.array(rows)
    y = df['label'].values
    X = np.nan_to_num(X, nan=0.0)

    # ── TASK 5: Train Model (Optimized Params) ────────────────
    print("\n[4/5] Training RandomForestClassifier (Optimized)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Optimized Classifier
    model = RandomForestClassifier(
        n_estimators=120,           # Increased slightly as requested
        max_depth=20,
        class_weight="balanced",   # Added for reliability
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    # ── TASK 6: Feature Importance Output ────────────────────
    print("\n[INFO] Top 5 Most Important Features:")
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:5]
    for i in indices:
        print(f"    {FEATURE_NAMES[i]}: {importances[i]:.4f}")

    # Evaluation
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)


    print("-" * 30)
    print(f"  Accuracy:  {acc*100:.2f}%")
    print(f"  Precision: {prec*100:.2f}%")
    print(f"  Recall:    {rec*100:.2f}%")
    print(f"  F1-Score:  {f1*100:.2f}%")
    print("-" * 30)

    # ── TASK 6: Save Model ────────────────────────────────────
    print("\n[5/5] Saving model resources...")
    os.makedirs("model", exist_ok=True)
    
    # Save as rf_model.pkl for compatibility with detector.py
    joblib.dump(model, "model/rf_model.pkl")
    joblib.dump(scaler, "model/scaler.pkl")
    joblib.dump(FEATURE_NAMES, "model/feature_names.pkl")
    
    print("  [SUCCESS] Saved: model/rf_model.pkl")
    print("  [SUCCESS] Saved: model/scaler.pkl")
    print("  [SUCCESS] Saved: model/feature_names.pkl")

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
