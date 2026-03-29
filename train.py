#!/usr/bin/env python
"""
PhishNet train.py - CORRECTED VERSION
Dataset: malicious_phish.csv (url, type)
Labels: benign=SAFE, phishing/malware/defacement=MALICIOUS
"""
import pandas as pd
import numpy as np
import joblib, os, sys, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix)
from imblearn.over_sampling import SMOTE

sys.path.insert(0, ".")
from src.features import extract_features, FEATURE_NAMES

print("=" * 60)
print("  PhishNet - Training Script")
print("=" * 60)

# ── STEP 1: Load ──────────────────────────────────────────────
print("\n[1/7] Loading dataset...")
DATA = None
for p in ["data/malicious_phish.csv",
          "data/malicious_phish (1).csv",
          "data/dataset.csv"]:
    if os.path.exists(p):
        DATA = p; break

if not DATA:
    print("  ERROR: CSV not found in data/ folder!")
    sys.exit(1)

df = pd.read_csv(DATA)
print(f"  Rows: {len(df):,}  Columns: {list(df.columns)}")

url_col   = "url"   if "url"   in df.columns else df.columns[0]
label_col = "type"  if "type"  in df.columns else df.columns[-1]

vc = df[label_col].value_counts()
print(f"\n  Label distribution:")
for lbl, cnt in vc.items():
    print(f"    {str(lbl):<20} {cnt:>8,}")

# ── STEP 2: Binary encode ─────────────────────────────────────
print("\n[2/7] Encoding labels...")
SAFE_LABELS = {"benign", "legitimate", "safe", "good"}
df["label_bin"] = df[label_col].astype(str).str.lower().str.strip().apply(
    lambda x: 0 if x in SAFE_LABELS else 1
)
n0 = (df["label_bin"] == 0).sum()
n1 = (df["label_bin"] == 1).sum()
print(f"  SAFE (0):      {n0:>8,}")
print(f"  MALICIOUS (1): {n1:>8,}")

# ── STEP 3: Features ──────────────────────────────────────────
FEAT_CACHE = "data/features_cache.csv"
print("\n[3/7] Extracting features...")

if os.path.exists(FEAT_CACHE):
    df_feat = pd.read_csv(FEAT_CACHE)
    if list(df_feat.columns) == FEATURE_NAMES + ["label_bin"] and len(df_feat) == len(df):
        print(f"  Using cache ({len(df_feat):,} rows)")
    else:
        print("  Cache mismatch — re-extracting...")
        os.remove(FEAT_CACHE)
        df_feat = None
else:
    df_feat = None

if df_feat is None:
    rows = []
    total = len(df)
    print(f"  Extracting {total:,} URLs...")
    for i, row in df.iterrows():
        try:
            f = extract_features(str(row[url_col]))
        except:
            f = [0.0] * len(FEATURE_NAMES)
        f.append(int(row["label_bin"]))
        rows.append(f)
        if i % 5000 == 0 and i > 0:
            print(f"    {i:>6,}/{total:,}  ({i/total*100:.0f}%)")

    df_feat = pd.DataFrame(rows, columns=FEATURE_NAMES + ["label_bin"])
    df_feat.to_csv(FEAT_CACHE, index=False)
    print(f"  Saved cache!")

# ── STEP 4: Check balance ─────────────────────────────────────
print("\n[4/7] Checking data balance...")
X = df_feat[FEATURE_NAMES].values.astype(float)
y = df_feat["label_bin"].values.astype(int)
X = np.nan_to_num(X, nan=0.0)

n_safe = (y == 0).sum()
n_mal  = (y == 1).sum()
print(f"  SAFE:      {n_safe:,}")
print(f"  MALICIOUS: {n_mal:,}")

# ── STEP 5: Train ─────────────────────────────────────────────
print("\n[5/7] Training...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# SMOTE on training set only
sm = SMOTE(random_state=42)
X_tr_s, y_tr_s = sm.fit_resample(X_train, y_train)

scaler  = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr_s)
X_te_sc = scaler.transform(X_test)

# Use class_weight to help with imbalance
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=2,
        class_weight={0: 1, 1: 2},   # penalise missing malicious more
        random_state=42,
        n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=15,
        class_weight={0:1, 1:2},
        random_state=42
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        random_state=42
    ),
}

trained, results = {}, {}
for name, model in models.items():
    print(f"  Training {name}...", end="  ", flush=True)
    model.fit(X_tr_sc, y_tr_s)
    yp = model.predict(X_te_sc)
    results[name] = {
        "acc":  round(accuracy_score(y_test, yp)*100, 2),
        "prec": round(precision_score(y_test, yp, zero_division=0)*100, 2),
        "rec":  round(recall_score(y_test, yp, zero_division=0)*100, 2),
        "f1":   round(f1_score(y_test, yp, zero_division=0)*100, 2),
        "pred": yp,
    }
    trained[name] = model
    print(f"Accuracy: {results[name]['acc']}%")

# ── STEP 6: Verify on known URLs ──────────────────────────────
print("\n[6/7] Verifying on real URLs...")
rf = trained["Random Forest"]

checks = [
    ("https://www.google.com",              0, "SAFE"),
    ("https://www.amazon.com",              0, "SAFE"),
    ("https://www.youtube.com",             0, "SAFE"),
    ("https://chatgpt.com",                 0, "SAFE"),
    ("https://www.instagram.com",           0, "SAFE"),
    ("https://www.linkedin.com",            0, "SAFE"),
    ("http://192.168.1.1/phishing",         1, "MALICIOUS"),
    ("http://free-paypal-verify.xyz/login", 1, "MALICIOUS"),
]

ok_count = 0
for url, exp, exp_name in checks:
    feats = extract_features(url)
    df_v  = pd.DataFrame([feats], columns=FEATURE_NAMES)
    sc_v  = scaler.transform(df_v)
    pred  = int(rf.predict(sc_v)[0])
    prob  = rf.predict_proba(sc_v)[0]
    label = "SAFE" if pred == 0 else "MALICIOUS"
    conf  = round(float(max(prob))*100, 1)
    ok    = pred == exp
    icon  = "✅" if ok else "❌"
    print(f"  {icon} {url[:45]:<45} -> {label} ({conf}%)")
    if ok:
        ok_count += 1

print(f"\n  {ok_count}/{len(checks)} verified correctly")

if ok_count < 5:
    print("\n  ⚠️  Still some wrong — but saving anyway.")
    print("  The model is the best we can do with URL structure alone.")

# ── STEP 7: Save + Charts ─────────────────────────────────────
print("\n[7/7] Saving model and generating charts...")
os.makedirs("model", exist_ok=True)
joblib.dump(trained["Random Forest"],     "model/rf_model.pkl")
joblib.dump(trained["Gradient Boosting"], "model/gb_model.pkl")
joblib.dump(scaler,                       "model/scaler.pkl")
joblib.dump(FEATURE_NAMES,                "model/feature_names.pkl")
print("  All model files saved!")

os.makedirs("static/charts", exist_ok=True)
C = ["#0D1B4B","#1565C0","#EA580C","#166534"]

# Accuracy chart
fig, ax = plt.subplots(figsize=(9, 5))
names = list(results.keys())
accs  = [results[n]["acc"] for n in names]
bars  = ax.bar(names, accs, color=C[:len(names)], width=0.5, zorder=3)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            f"{acc}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.axhline(95, color="#EA580C", linestyle="--", linewidth=1.5, label="95% target")
ax.set_ylim(70, 105); ax.set_ylabel("Accuracy (%)"); ax.legend()
ax.set_title("Model Accuracy Comparison — PhishNet", fontsize=13, fontweight="bold")
ax.grid(axis="y", alpha=0.3); plt.xticks(rotation=20, ha="right", fontsize=9)
plt.tight_layout()
plt.savefig("static/charts/accuracy.png", dpi=150, bbox_inches="tight"); plt.close()

# Confusion matrix
rf_pred = results["Random Forest"]["pred"]
cm_rf   = confusion_matrix(y_test, rf_pred)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Legitimate","Malicious"],
            yticklabels=["Legitimate","Malicious"])
ax.set_title("Random Forest — Confusion Matrix", fontsize=12, fontweight="bold")
ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
plt.tight_layout()
plt.savefig("static/charts/confusion_rf.png", dpi=150, bbox_inches="tight"); plt.close()

# All metrics chart
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(names)); w = 0.2
for i,(m,lbl,col) in enumerate(zip(
        ["acc","prec","rec","f1"],
        ["Accuracy","Precision","Recall","F1"], C)):
    ax.bar(x+i*w, [results[n][m] for n in names], w, label=lbl, color=col, zorder=3)
ax.set_xticks(x+w*1.5); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
ax.set_ylim(70, 105); ax.legend(fontsize=9)
ax.set_title("All Metrics — PhishNet", fontsize=13, fontweight="bold")
ax.grid(axis="y", alpha=0.3); plt.tight_layout()
plt.savefig("static/charts/all_metrics.png", dpi=150, bbox_inches="tight"); plt.close()

# Dataset pie
fig, ax = plt.subplots(figsize=(5, 5))
ax.pie([n_safe, n_mal],
       labels=[f"Legitimate\n{round(n_safe/(n_safe+n_mal)*100,1)}%",
               f"Malicious\n{round(n_mal/(n_safe+n_mal)*100,1)}%"],
       colors=["#166534","#991B1B"], autopct="%1.1f%%",
       startangle=140, textprops={"fontsize":10})
ax.set_title("Dataset Distribution", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("static/charts/dataset_pie.png", dpi=150, bbox_inches="tight"); plt.close()

# Feature importance
fi  = trained["Random Forest"].feature_importances_
idx = np.argsort(fi)[-15:]
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh([FEATURE_NAMES[i] for i in idx], fi[idx], color="#1565C0")
ax.set_title("Feature Importances — Random Forest", fontsize=12, fontweight="bold")
ax.set_xlabel("Importance Score"); plt.tight_layout()
plt.savefig("static/charts/feature_importance.png", dpi=150, bbox_inches="tight"); plt.close()

print("\n" + "="*60)
print(f"  {'Model':<25} {'Accuracy':>9} {'Precision':>10} {'F1':>8}")
print("  " + "-"*55)
for n, r in results.items():
    print(f"  {n:<25} {r['acc']:>8.2f}%  {r['prec']:>9.2f}%  {r['f1']:>7.2f}%")
print("="*60)
print(f"\n  ✅  TRAINING COMPLETE!")
print(f"  Best model: Random Forest — {results['Random Forest']['acc']}%")
print("="*60)
