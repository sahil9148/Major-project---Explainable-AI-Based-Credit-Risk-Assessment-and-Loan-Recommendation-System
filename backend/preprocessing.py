"""
preprocessing.py
-----------------
Shared feature engineering utilities used by BOTH the training pipeline
(train.py) and the inference API (main.py). Keeping this logic in one
place prevents "train/serve skew" -- a common source of silent bugs in
production ML systems, where the API preprocesses a request slightly
differently than the training script preprocessed its data, and the
model ends up scoring garbage without ever throwing an error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = [
    "age",
    "annual_income",
    "loan_amount",
    "credit_score",
    "employment_length_years",
    "debt_to_income_ratio",
    "num_open_credit_lines",
    "num_credit_inquiries_last_6m",
    "credit_utilization_rate",
    "past_defaults",
]

HOME_OWNERSHIP_CATEGORIES = ["RENT", "MORTGAGE", "OWN"]
LOAN_PURPOSE_CATEGORIES = [
    "debt_consolidation",
    "home_improvement",
    "education",
    "business",
    "medical",
    "other",
]

CATEGORICAL_FEATURES = {
    "home_ownership": HOME_OWNERSHIP_CATEGORIES,
    "loan_purpose": LOAN_PURPOSE_CATEGORIES,
}


def generate_synthetic_dataset(n_samples: int = 8000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic credit-application dataset with a *plausible*,
    learnable relationship between applicant attributes and default risk.

    This is NOT real financial data -- it exists purely so the pipeline has
    something realistic to train and explain against. Swap this out for a
    real, licensed/anonymized credit dataset before this touches anything
    resembling a real lending decision.
    """
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 71, n_samples)
    annual_income = rng.lognormal(mean=10.9, sigma=0.45, size=n_samples)
    annual_income = np.clip(annual_income, 15_000, 280_000)

    employment_length_years = np.minimum(
        rng.integers(0, 41, n_samples), np.clip(age - 18, 0, None)
    )

    credit_score = np.clip(rng.normal(660, 90, n_samples), 300, 850).round()

    loan_amount = rng.uniform(1_000, 50_000, n_samples)
    debt_to_income_ratio = rng.beta(2, 5, n_samples) * 0.65
    num_open_credit_lines = rng.integers(0, 21, n_samples)
    num_credit_inquiries_last_6m = np.clip(rng.poisson(1.2, n_samples), 0, 10)
    credit_utilization_rate = rng.beta(2, 3, n_samples)
    past_defaults = np.clip(rng.poisson(0.3, n_samples), 0, 5)

    home_ownership = rng.choice(
        HOME_OWNERSHIP_CATEGORIES, size=n_samples, p=[0.40, 0.45, 0.15]
    )
    loan_purpose = rng.choice(
        LOAN_PURPOSE_CATEGORIES,
        size=n_samples,
        p=[0.32, 0.18, 0.14, 0.16, 0.10, 0.10],
    )

    # Hand-tuned "ground truth" latent relationship, calibrated so income,
    # credit history, and utilization dominate the decision -- similar to
    # real underwriting heuristics -- while leaving enough stochastic noise
    # (via the binomial draw below) that the problem isn't trivially
    # separable, which is what makes SHAP's explanations interesting.
    logit = (
        -0.000020 * annual_income
        + 3.000 * debt_to_income_ratio
        - 0.006 * credit_score
        + 0.900 * past_defaults
        + 1.500 * credit_utilization_rate
        + 0.150 * num_credit_inquiries_last_6m
        - 0.030 * employment_length_years
        + 0.000020 * loan_amount
        + 1.500
    )
    prob_default = 1 / (1 + np.exp(-logit))
    default_label = rng.binomial(1, prob_default)

    return pd.DataFrame(
        {
            "age": age,
            "annual_income": annual_income.round(2),
            "loan_amount": loan_amount.round(2),
            "credit_score": credit_score,
            "employment_length_years": employment_length_years,
            "debt_to_income_ratio": debt_to_income_ratio.round(4),
            "num_open_credit_lines": num_open_credit_lines,
            "num_credit_inquiries_last_6m": num_credit_inquiries_last_6m,
            "credit_utilization_rate": credit_utilization_rate.round(4),
            "past_defaults": past_defaults,
            "home_ownership": home_ownership,
            "loan_purpose": loan_purpose,
            "default": default_label,
        }
    )


def build_model_matrix(
    df: pd.DataFrame, feature_columns: list[str] | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """One-hot encode categorical columns and return (X, feature_columns).

    If `feature_columns` is provided (inference time), the resulting frame
    is reindexed to exactly match it -- adding any missing dummy columns as
    zeros and dropping unexpected ones -- so a single applicant record lines
    up with the exact columns the model was trained on, even though a lone
    applicant only ever produces one non-zero dummy per categorical field.
    """
    encoded = pd.get_dummies(
        df, columns=list(CATEGORICAL_FEATURES.keys()), dtype=int
    )

    if feature_columns is None:
        feature_columns = sorted(encoded.columns)

    encoded = encoded.reindex(columns=feature_columns, fill_value=0)
    return encoded, feature_columns
