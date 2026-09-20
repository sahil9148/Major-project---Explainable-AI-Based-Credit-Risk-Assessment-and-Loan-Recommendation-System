"""
schemas.py
----------
Pydantic request/response models for the credit risk API. These are the
single source of truth for what the API accepts and returns -- FastAPI
uses them to validate requests, serialize responses, and auto-generate
the OpenAPI docs at /docs.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class HomeOwnership(str, Enum):
    RENT = "RENT"
    MORTGAGE = "MORTGAGE"
    OWN = "OWN"


class LoanPurpose(str, Enum):
    DEBT_CONSOLIDATION = "debt_consolidation"
    HOME_IMPROVEMENT = "home_improvement"
    EDUCATION = "education"
    BUSINESS = "business"
    MEDICAL = "medical"
    OTHER = "other"


class ApplicantData(BaseModel):
    """Raw applicant data submitted by the frontend."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 34,
                "annual_income": 78000,
                "loan_amount": 15000,
                "credit_score": 705,
                "employment_length_years": 6,
                "debt_to_income_ratio": 0.28,
                "num_open_credit_lines": 5,
                "num_credit_inquiries_last_6m": 1,
                "credit_utilization_rate": 0.32,
                "past_defaults": 0,
                "home_ownership": "MORTGAGE",
                "loan_purpose": "debt_consolidation",
            }
        }
    )

    age: int = Field(..., ge=18, le=100, description="Applicant age in years")
    annual_income: float = Field(..., gt=0, description="Gross annual income, USD")
    loan_amount: float = Field(..., gt=0, description="Requested loan amount, USD")
    credit_score: int = Field(..., ge=300, le=850, description="FICO-style credit score")
    employment_length_years: float = Field(..., ge=0, le=60)
    debt_to_income_ratio: float = Field(
        ..., ge=0, le=1, description="Total monthly debt payments / monthly income"
    )
    num_open_credit_lines: int = Field(..., ge=0, le=100)
    num_credit_inquiries_last_6m: int = Field(..., ge=0, le=50)
    credit_utilization_rate: float = Field(..., ge=0, le=1)
    past_defaults: int = Field(..., ge=0, le=50)
    home_ownership: HomeOwnership
    loan_purpose: LoanPurpose


class FeatureContribution(BaseModel):
    feature: str
    value: Union[float, str]
    shap_value: float = Field(..., description="Contribution in log-odds space")
    direction: Literal["increases_risk", "decreases_risk"]


class CounterfactualSuggestion(BaseModel):
    feature: str = Field(..., description="Which feature to change, or 'combined' for a multi-feature suggestion")
    description: str
    new_risk_score: float
    new_tier: Literal["LOW", "MODERATE", "HIGH", "VERY_HIGH"]


class RiskAssessment(BaseModel):
    # model_version legitimately starts with "model_" (an ML model version,
    # not a Pydantic model) -- silence Pydantic's protected-namespace warning.
    model_config = ConfigDict(protected_namespaces=())

    risk_score: float = Field(..., description="Predicted probability of default, 0-1")
    risk_tier: Literal["LOW", "MODERATE", "HIGH", "VERY_HIGH"]
    recommendation: Literal[
        "APPROVE", "APPROVE_WITH_CONDITIONS", "MANUAL_REVIEW", "DENY"
    ]
    recommended_interest_rate_pct: Union[float, None] = None
    recommended_loan_amount: Union[float, None] = None
    base_value: float = Field(..., description="Model's baseline log-odds before SHAP adjustments")
    top_contributions: list[FeatureContribution]
    narrative: str = Field(..., description="Plain-English explanation of the decision")
    counterfactuals: list[CounterfactualSuggestion] = Field(
        default_factory=list,
        description="Smallest realistic changes that would improve the applicant's risk tier; empty if already LOW risk",
    )
    model_version: str


# ---------------------------------------------------------------------------
# Global model insights (GET /insights) -- properties of the model as a
# whole, computed once at training time, not per-applicant.
# ---------------------------------------------------------------------------


class ROCPoint(BaseModel):
    fpr: float
    tpr: float


class CalibrationPoint(BaseModel):
    mean_predicted: float
    observed_frequency: float


class ConfusionMatrixCounts(BaseModel):
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int


class GlobalFeatureImportance(BaseModel):
    feature: str
    mean_abs_shap: float


class ModelInsights(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    roc_auc: float
    accuracy: float
    n_test_samples: int
    trained_at: str
    roc_curve: list[ROCPoint]
    calibration_curve: list[CalibrationPoint]
    confusion_matrix: ConfusionMatrixCounts
    global_feature_importance: list[GlobalFeatureImportance]


# ---------------------------------------------------------------------------
# Fairness audit (GET /fairness)
# ---------------------------------------------------------------------------


class FairnessGroup(BaseModel):
    group: str
    n: int
    selection_rate: float
    actual_default_rate: float
    mean_predicted_risk: float
    true_positive_rate: Union[float, None]


class FairnessReport(BaseModel):
    groups: list[FairnessGroup]
    four_fifths_ratio: Union[float, None]
    four_fifths_pass: Union[bool, None]
    methodology: str
