"""
main.py
-------
FastAPI service exposing the credit risk model for real-time inference.

Endpoints
---------
GET  /health        liveness check
GET  /model/info    metadata about the currently loaded model
GET  /insights       global model evaluation: ROC curve, calibration,
                      confusion matrix, global SHAP importance
GET  /fairness       age-group fairness audit (selection rate, TPR,
                      four-fifths rule)
POST /predict         score a single applicant: SHAP-based explanations,
                      a loan recommendation, a plain-English narrative,
                      and "what would it take" counterfactual suggestions
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from counterfactual import find_counterfactuals
from narrative import generate_narrative
from preprocessing import CATEGORICAL_FEATURES, build_model_matrix
from schemas import (
    ApplicantData,
    CalibrationPoint,
    ConfusionMatrixCounts,
    CounterfactualSuggestion,
    FairnessReport,
    FeatureContribution,
    GlobalFeatureImportance,
    ModelInsights,
    ROCPoint,
    RiskAssessment,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("credit-risk-api")

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "xgb_credit_model.json"
METADATA_PATH = ROOT / "models" / "metadata.json"
MODEL_VERSION = "xgb-credit-risk-v1"

ml_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise RuntimeError(
            "Model artifacts not found. Run `python train.py` before starting the API."
        )

    logger.info("Loading model from %s", MODEL_PATH)
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))

    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    explainer = shap.TreeExplainer(model)

    ml_state["model"] = model
    ml_state["explainer"] = explainer
    ml_state["feature_columns"] = metadata["feature_columns"]
    ml_state["metadata"] = metadata
    logger.info("Model loaded with %d features.", len(metadata["feature_columns"]))

    yield
    ml_state.clear()


app = FastAPI(
    title="Explainable Credit Risk API",
    description="XGBoost + SHAP powered credit risk scoring and loan recommendation service.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your deployed frontend origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Business-rules layer. Deliberately kept separate from the ML model --
# lenders tune approval thresholds and pricing constantly without ever
# wanting to retrain (or re-validate) the underlying risk model.
# ---------------------------------------------------------------------------


def score_to_tier(score: float) -> str:
    if score < 0.15:
        return "LOW"
    if score < 0.40:
        return "MODERATE"
    if score < 0.70:
        return "HIGH"
    return "VERY_HIGH"


def build_recommendation(score: float, requested_amount: float) -> dict:
    if score < 0.15:
        return {
            "recommendation": "APPROVE",
            "recommended_interest_rate_pct": round(5.5 + score * 10, 2),
            "recommended_loan_amount": requested_amount,
        }
    if score < 0.40:
        return {
            "recommendation": "APPROVE_WITH_CONDITIONS",
            "recommended_interest_rate_pct": round(8.0 + score * 12, 2),
            "recommended_loan_amount": round(requested_amount * 0.85, 2),
        }
    if score < 0.70:
        return {
            "recommendation": "MANUAL_REVIEW",
            "recommended_interest_rate_pct": None,
            "recommended_loan_amount": round(requested_amount * 0.5, 2),
        }
    return {
        "recommendation": "DENY",
        "recommended_interest_rate_pct": None,
        "recommended_loan_amount": None,
    }


def humanize_contributions(
    feature_columns: list[str],
    values: list[float],
    shap_row: list[float],
) -> list[FeatureContribution]:
    """Collapse one-hot dummy columns back into a single contribution per
    logical feature (e.g. combine home_ownership_RENT / _MORTGAGE / _OWN
    into one 'home_ownership' entry) by summing their SHAP values -- the
    mathematically valid way to regroup SHAP attributions, and far more
    readable for a non-technical reviewer than a wall of 0/1 dummy columns.
    """
    dummy_lookup = {
        f"{cat}_{option}": (cat, option)
        for cat, options in CATEGORICAL_FEATURES.items()
        for option in options
    }

    grouped: dict[str, dict[str, Any]] = {}
    for feat, val, sv in zip(feature_columns, values, shap_row):
        if feat in dummy_lookup:
            base_feat, option = dummy_lookup[feat]
            entry = grouped.setdefault(base_feat, {"shap": 0.0, "value": None})
            entry["shap"] += sv
            if val == 1:
                entry["value"] = option
        else:
            grouped[feat] = {"shap": sv, "value": val}

    contributions = [
        FeatureContribution(
            feature=name,
            value=data["value"] if data["value"] is not None else "n/a",
            shap_value=round(float(data["shap"]), 5),
            direction="increases_risk" if data["shap"] > 0 else "decreases_risk",
        )
        for name, data in grouped.items()
    ]
    contributions.sort(key=lambda c: abs(c.shap_value), reverse=True)
    return contributions


@app.get("/")
async def root() -> dict:
    return {
        "service": "Explainable Credit Risk API",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model_loaded": "model" in ml_state}


@app.get("/model/info")
async def model_info() -> dict:
    if "metadata" not in ml_state:
        raise HTTPException(503, "Model not loaded")
    meta = ml_state["metadata"]
    return {
        "model_version": MODEL_VERSION,
        "trained_at": meta.get("trained_at"),
        "n_training_samples": meta.get("n_samples"),
        "roc_auc": meta.get("metrics", {}).get("roc_auc"),
        "top_global_features": list(meta.get("global_shap_importance", {}).items())[:8],
    }


@app.get("/insights", response_model=ModelInsights)
async def insights() -> ModelInsights:
    """Properties of the model as a whole -- computed once in train.py,
    served statically here rather than recomputed per request."""
    if "metadata" not in ml_state:
        raise HTTPException(503, "Model not loaded")
    meta = ml_state["metadata"]
    missing = [
        k for k in ("roc_curve", "calibration_curve", "confusion_matrix") if k not in meta
    ]
    if missing:
        raise HTTPException(
            503,
            f"This model was trained before {', '.join(missing)} were added -- "
            "re-run `python train.py` to regenerate metadata.json.",
        )

    return ModelInsights(
        roc_auc=meta["metrics"]["roc_auc"],
        accuracy=meta["metrics"]["accuracy"],
        n_test_samples=meta.get("n_test_samples", 0),
        trained_at=meta["trained_at"],
        roc_curve=[ROCPoint(**p) for p in meta["roc_curve"]],
        calibration_curve=[CalibrationPoint(**p) for p in meta["calibration_curve"]],
        confusion_matrix=ConfusionMatrixCounts(**meta["confusion_matrix"]),
        global_feature_importance=[
            GlobalFeatureImportance(feature=k, mean_abs_shap=v)
            for k, v in meta.get("global_shap_importance", {}).items()
        ],
    )


@app.get("/fairness", response_model=FairnessReport)
async def fairness() -> FairnessReport:
    """Age-group fairness audit -- computed once in train.py (see
    fairness.py for methodology), served statically here."""
    if "metadata" not in ml_state:
        raise HTTPException(503, "Model not loaded")
    meta = ml_state["metadata"]
    report = meta.get("fairness_report")
    if report is None:
        raise HTTPException(
            503,
            "This model was trained before the fairness audit was added -- "
            "re-run `python train.py` to regenerate metadata.json.",
        )
    return FairnessReport(**report)


@app.post("/predict", response_model=RiskAssessment)
async def predict(applicant: ApplicantData) -> RiskAssessment:
    if "model" not in ml_state:
        raise HTTPException(503, "Model not loaded")

    model: xgb.XGBClassifier = ml_state["model"]
    explainer = ml_state["explainer"]
    feature_columns: list[str] = ml_state["feature_columns"]

    # mode="json" guarantees Enum fields serialize to plain strings
    # (e.g. "MORTGAGE") regardless of Python version, instead of the
    # Enum member's repr -- which would silently break get_dummies().
    raw = pd.DataFrame([applicant.model_dump(mode="json")])
    X, _ = build_model_matrix(raw, feature_columns=feature_columns)

    risk_score = float(model.predict_proba(X)[0, 1])

    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):  # defensive: SHAP/model version differences
        shap_values = shap_values[-1]
    shap_row = np.asarray(shap_values)[0].tolist()

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[-1]

    contributions = humanize_contributions(feature_columns, X.iloc[0].tolist(), shap_row)
    decision = build_recommendation(risk_score, applicant.loan_amount)
    tier = score_to_tier(risk_score)

    counterfactuals = find_counterfactuals(
        model=model,
        X_base=X,
        current_tier=tier,
    )

    narrative = generate_narrative(
        risk_score=risk_score,
        risk_tier=tier,
        recommendation=decision["recommendation"],
        top_contributions=contributions[:6],
    )

    return RiskAssessment(
        risk_score=round(risk_score, 5),
        risk_tier=tier,
        base_value=round(float(base_value), 5),
        top_contributions=contributions[:8],
        narrative=narrative,
        counterfactuals=[
            CounterfactualSuggestion(
                feature=cf.feature,
                description=cf.description,
                new_risk_score=cf.new_risk_score,
                new_tier=cf.new_tier,
            )
            for cf in counterfactuals
        ],
        model_version=MODEL_VERSION,
        **decision,
    )
