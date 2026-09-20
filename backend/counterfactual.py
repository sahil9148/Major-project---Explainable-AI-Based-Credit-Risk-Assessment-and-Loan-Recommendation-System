"""
counterfactual.py
------------------
"What would it take?" search: given a scored applicant, find the smallest
realistic change to *actionable* features that would move them to a better
risk tier. This is what turns the system from "here's why you were denied"
into "here's what to do about it" -- the difference between descriptive
and actionable explainability.

Deliberately excludes features an applicant can't realistically act on in
the short term (age, past defaults, income) and only searches features
where "do less of this" is legitimate, achievable financial advice.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from preprocessing import build_model_matrix

# Features we're willing to suggest changing. Order is just the order
# single-feature suggestions are attempted in.
ACTIONABLE_FEATURES: list[str] = [
    "credit_utilization_rate",
    "debt_to_income_ratio",
    "loan_amount",
    "num_credit_inquiries_last_6m",
]

# We won't suggest reducing a feature below this floor even if the search
# would technically keep improving the score.
FEATURE_FLOORS = {
    "credit_utilization_rate": 0.05,
    "debt_to_income_ratio": 0.02,
    "loan_amount": 1000.0,
    "num_credit_inquiries_last_6m": 0.0,
}

_TIER_RANK = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "VERY_HIGH": 3}


def _tier_from_score(score: float) -> str:
    if score < 0.15:
        return "LOW"
    if score < 0.40:
        return "MODERATE"
    if score < 0.70:
        return "HIGH"
    return "VERY_HIGH"


def _score_for(model, feature_columns: list[str], row: pd.DataFrame) -> float:
    X, _ = build_model_matrix(row, feature_columns=feature_columns)
    return float(model.predict_proba(X)[0, 1])


@dataclass
class Counterfactual:
    feature: str
    current_value: float
    suggested_value: float
    new_risk_score: float
    new_tier: str
    description: str


def _describe(feature: str, current: float, suggested: float, new_tier: str) -> str:
    labels = {
        "credit_utilization_rate": ("credit utilization", lambda v: f"{v * 100:.0f}%"),
        "debt_to_income_ratio": ("debt-to-income ratio", lambda v: f"{v * 100:.0f}%"),
        "loan_amount": ("requested loan amount", lambda v: f"${v:,.0f}"),
        "num_credit_inquiries_last_6m": ("recent credit inquiries", lambda v: f"{v:.0f}"),
    }
    name, fmt = labels[feature]
    return (
        f"Reducing {name} from {fmt(current)} to {fmt(suggested)} "
        f"would move this applicant to {new_tier} risk."
    )


def _search_single_feature(
    model,
    feature_columns: list[str],
    base_row: pd.DataFrame,
    feature: str,
    current_tier: str,
) -> "Counterfactual | None":
    """Binary-search a reduction in `feature` for the smallest change that
    improves the applicant by at least one risk tier."""
    current_value = float(base_row.iloc[0][feature])
    floor = FEATURE_FLOORS.get(feature, 0.0)
    if current_value <= floor:
        return None

    lo, hi = floor, current_value  # invariant: lo improves (if anything does), hi does not
    best = None

    trial = base_row.copy()
    trial[feature] = trial[feature].astype(float)  # binary search needs fractional
    # midpoints even for integer-valued columns like num_credit_inquiries_last_6m --
    # without this cast, pandas silently (soon: not-so-silently) truncates them.

    for _ in range(20):  # 20 bisections is far more precision than this advice needs
        mid = (lo + hi) / 2
        trial.at[trial.index[0], feature] = mid
        score = _score_for(model, feature_columns, trial)
        tier = _tier_from_score(score)
        if _TIER_RANK[tier] < _TIER_RANK[current_tier]:
            best = (mid, score, tier)
            lo = mid  # this works -- see if an even smaller change also works
        else:
            hi = mid  # doesn't work yet -- need more reduction

    if best is None:
        return None

    suggested_value, new_score, new_tier = best
    return Counterfactual(
        feature=feature,
        current_value=round(current_value, 4),
        suggested_value=round(suggested_value, 4),
        new_risk_score=round(new_score, 5),
        new_tier=new_tier,
        description=_describe(feature, current_value, suggested_value, new_tier),
    )


def _search_combined(
    model,
    feature_columns: list[str],
    base_row: pd.DataFrame,
    current_tier: str,
) -> "Counterfactual | None":
    """One 'balanced' suggestion: shrink every actionable feature by the
    same percentage simultaneously. Often more realistic advice than one
    large single-feature change."""
    current_values = {f: float(base_row.iloc[0][f]) for f in ACTIONABLE_FEATURES}

    lo, hi = 0.0, 1.0  # fraction to reduce every actionable feature by
    best = None

    trial = base_row.copy()
    for f in ACTIONABLE_FEATURES:
        trial[f] = trial[f].astype(float)

    for _ in range(20):
        mid = (lo + hi) / 2
        for f in ACTIONABLE_FEATURES:
            floor = FEATURE_FLOORS.get(f, 0.0)
            trial.at[trial.index[0], f] = max(current_values[f] * (1 - mid), floor)
        score = _score_for(model, feature_columns, trial)
        tier = _tier_from_score(score)
        if _TIER_RANK[tier] < _TIER_RANK[current_tier]:
            best = (mid, score, tier)
            hi = mid  # this works -- see if a smaller combined change also works
        else:
            lo = mid

    if best is None:
        return None

    fraction, new_score, new_tier = best
    pct = round(fraction * 100)
    return Counterfactual(
        feature="combined",
        current_value=0.0,
        suggested_value=fraction,
        new_risk_score=round(new_score, 5),
        new_tier=new_tier,
        description=(
            f"A {pct}% reduction spread across utilization, debt-to-income, "
            f"loan amount, and recent inquiries together would move this "
            f"applicant to {new_tier} risk -- often more realistic than one "
            f"large single change."
        ),
    )


def find_counterfactuals(
    model,
    feature_columns: list[str],
    applicant_row: pd.DataFrame,
    current_tier: str,
    max_suggestions: int = 3,
) -> list[Counterfactual]:
    if current_tier == "LOW":
        return []  # already the best tier -- nothing to suggest

    suggestions: list[Counterfactual] = []
    for feature in ACTIONABLE_FEATURES:
        cf = _search_single_feature(model, feature_columns, applicant_row, feature, current_tier)
        if cf is not None:
            suggestions.append(cf)

    combined = _search_combined(model, feature_columns, applicant_row, current_tier)
    if combined is not None:
        suggestions.append(combined)

    def relative_change(cf: Counterfactual) -> float:
        if cf.feature == "combined":
            return cf.suggested_value
        if cf.current_value == 0:
            return 1.0
        return abs(cf.current_value - cf.suggested_value) / cf.current_value

    suggestions.sort(key=relative_change)  # smallest, easiest changes first
    return suggestions[:max_suggestions]
