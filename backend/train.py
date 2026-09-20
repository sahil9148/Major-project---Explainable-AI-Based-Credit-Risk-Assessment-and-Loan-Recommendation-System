"""
train.py
--------
End-to-end training pipeline for the credit risk model:

    1. Generate a synthetic credit-application dataset.
    2. Split into train/test sets.
    3. Train a gradient-boosted tree classifier (XGBoost) to predict
       probability of default.
    4. Evaluate on the held-out test set: ROC-AUC, accuracy, full report,
       plus a downsampled ROC curve, a calibration curve, and a confusion
       matrix -- everything the /insights endpoint serves.
    5. Fit a SHAP TreeExplainer for post-hoc explainability and compute
       global feature importance.
    6. Run an age-group fairness audit (see fairness.py) -- everything the
       /fairness endpoint serves.
    7. Persist the model + feature schema + metrics + insights + fairness
       report to disk so the FastAPI service (main.py) can load them at
       startup and serve them without recomputing anything per request.

Run with:  python train.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from fairness import compute_fairness_report
from preprocessing import build_model_matrix, generate_synthetic_dataset

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

N_SAMPLES = 8_000
RANDOM_STATE = 42
TEST_SIZE = 0.2


def main() -> None:
    print("Step 1/6 -- generating synthetic credit dataset...")
    df = generate_synthetic_dataset(n_samples=N_SAMPLES, seed=RANDOM_STATE)
    df.to_csv(DATA_DIR / "synthetic_credit_data.csv", index=False)
    print(f"  saved {len(df):,} rows -> data/synthetic_credit_data.csv")

    y = df.pop("default")
    X, feature_columns = build_model_matrix(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print("Step 2/6 -- training XGBoost classifier...")
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    start = time.time()
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    print(f"  trained in {time.time() - start:.2f}s")

    print("Step 3/6 -- evaluating on held-out test set...")
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, proba)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)

    print(f"  ROC-AUC:  {auc:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))

    # Downsample the ROC curve to a fixed 40 points on an evenly-spaced FPR
    # grid -- roc_curve() returns one point per unique threshold, which can
    # be thousands of points and is overkill (and slow to ship) for a chart.
    fpr_raw, tpr_raw, _ = roc_curve(y_test, proba)
    fpr_grid = np.linspace(0, 1, 40)
    tpr_grid = np.interp(fpr_grid, fpr_raw, tpr_raw)
    roc_points = [
        {"fpr": round(float(f), 4), "tpr": round(float(t), 4)}
        for f, t in zip(fpr_grid, tpr_grid)
    ]

    prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=10)
    calibration_points = [
        {"mean_predicted": round(float(p), 4), "observed_frequency": round(float(t), 4)}
        for p, t in zip(prob_pred, prob_true)
    ]

    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    confusion = {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

    print("Step 4/6 -- running age-group fairness audit...")
    fairness_report = compute_fairness_report(
        ages=X_test["age"].values, y_true=y_test.values, y_proba=proba
    )
    print(f"  four-fifths ratio: {fairness_report['four_fifths_ratio']}")
    print(f"  four-fifths pass:  {fairness_report['four_fifths_pass']}")
    for g in fairness_report["groups"]:
        print(f"    {g['group']:<8} n={g['n']:<5} selection_rate={g['selection_rate']}")

    print("Step 5/6 -- fitting SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    shap_sample = explainer.shap_values(X_test.iloc[:200])
    if isinstance(shap_sample, list):  # defensive: SHAP/model version differences
        shap_sample = shap_sample[-1]
    mean_abs_shap = np.abs(shap_sample).mean(axis=0)
    global_importance = (
        pd.Series(mean_abs_shap, index=feature_columns)
        .sort_values(ascending=False)
        .to_dict()
    )
    print("  top 5 globally important features:")
    for name, val in list(global_importance.items())[:5]:
        print(f"    {name:<32} {val:.4f}")

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[-1]

    print("Step 6/6 -- saving model artifacts...")
    model.save_model(str(MODEL_DIR / "xgb_credit_model.json"))

    trained_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    metadata = {
        "feature_columns": feature_columns,
        "trained_at": trained_at,
        "n_samples": N_SAMPLES,
        "n_test_samples": int(len(y_test)),
        "metrics": {
            "roc_auc": float(auc),
            "accuracy": float(acc),
            "classification_report": report,
        },
        "global_shap_importance": {k: float(v) for k, v in global_importance.items()},
        "base_value": float(base_value),
        "roc_curve": roc_points,
        "calibration_curve": calibration_points,
        "confusion_matrix": confusion,
        "fairness_report": fairness_report,
    }
    with open(MODEL_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  model    -> {MODEL_DIR / 'xgb_credit_model.json'}")
    print(f"  metadata -> {MODEL_DIR / 'metadata.json'}")
    print("Done.")


if __name__ == "__main__":
    main()
