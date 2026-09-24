"""
counterfactual.py
------------------
"What would it take?" search: given a scored applicant, find the smallest
realistic change to *actionable* features that would move them to a better
risk tier.

Deliberately excludes features an applicant can't realistically act on in
the short term (age, past defaults, income) and only searches features
where "do less of this" is legitimate, achievable financial advice.

Performance note. This runs 5 independent binary searches per request
(one per actionable feature, plus one combined search). Profiling showed
XGBoost's predict_proba() has ~4-6ms of *fixed* per-call overhead that's
almost independent of batch size (60 rows in one call costs about the
same as 1 row). So all 5 searches run in lockstep: at each of the 12
bisection rounds, one 5-row batch (one candidate per search) is scored in
a single predict_proba() call, instead of 5 separate calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

ACTIONABLE_FEATURES: list[str] = [
    "credit_utilization_rate",
    "debt_to_income_ratio",
    "loan_amount",
    "num_credit_inquiries_last_6m",
]

FEATURE_FLOORS = {
    "credit_utilization_rate": 0.05,
    "debt_to_income_ratio": 0.02,
    "loan_amount": 1000.0,
    "num_credit_inquiries_last_6m": 0.0,
}

_TIER_RANK = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "VERY_HIGH": 3}
SEARCH_ITERATIONS = 12


def _tier_from_score(score: float) -> str:
    if score < 0.15:
        return "LOW"
    if score < 0.40:
        return "MODERATE"
    if score < 0.70:
        return "HIGH"
    return "VERY_HIGH"


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


def _describe_combined(fraction: float, new_tier: str) -> str:
    pct = round(fraction * 100)
    return (
        f"A {pct}% reduction spread across utilization, debt-to-income, "
        f"loan amount, and recent inquiries together would move this "
        f"applicant to {new_tier} risk -- often more realistic than one "
        f"large single change."
    )


class _SearchState:
    def __init__(self, name: str, kind: Literal["single", "combined"], lo: float, hi: float):
        self.name = name
        self.kind = kind
        self.lo = lo
        self.hi = hi
        self.best: tuple[float, float, str] | None = None

    def candidate(self) -> float:
        return (self.lo + self.hi) / 2

    def update(self, mid: float, score: float, current_tier: str) -> None:
        tier = _tier_from_score(score)
        improved = _TIER_RANK[tier] < _TIER_RANK[current_tier]
        if improved:
            self.best = (mid, score, tier)
        if self.kind == "single":
            if improved:
                self.lo = mid
            else:
                self.hi = mid
        else:
            if improved:
                self.hi = mid
            else:
                self.lo = mid


def _apply_candidate(template: pd.DataFrame, state: "_SearchState", mid: float, combined_current: dict) -> pd.DataFrame:
    row = template.copy()
    if state.kind == "single":
        row.at[row.index[0], state.name] = mid
    else:
        for f in ACTIONABLE_FEATURES:
            floor = FEATURE_FLOORS.get(f, 0.0)
            row.at[row.index[0], f] = max(combined_current[f] * (1 - mid), floor)
    return row


def find_counterfactuals(
    model,
    X_base: pd.DataFrame,
    current_tier: str,
    max_suggestions: int = 3,
) -> list[Counterfactual]:
    """`X_base` is the already one-hot-encoded, feature_columns-aligned row
    for this applicant -- pass the same frame already built for the main
    prediction rather than raw applicant data, so this doesn't redundantly
    re-encode it."""
    if current_tier == "LOW":
        return []

    template = X_base.copy()
    for f in ACTIONABLE_FEATURES:
        template[f] = template[f].astype(float)

    combined_current = {f: float(X_base.iloc[0][f]) for f in ACTIONABLE_FEATURES}

    searches: list[_SearchState] = []
    for feature in ACTIONABLE_FEATURES:
        current_value = float(X_base.iloc[0][feature])
        floor = FEATURE_FLOORS.get(feature, 0.0)
        if current_value > floor:
            searches.append(_SearchState(feature, "single", floor, current_value))
    searches.append(_SearchState("combined", "combined", 0.0, 1.0))

    for _ in range(SEARCH_ITERATIONS):
        mids = [s.candidate() for s in searches]
        batch = pd.concat(
            [_apply_candidate(template, s, mid, combined_current) for s, mid in zip(searches, mids)],
            ignore_index=True,
        )
        scores = model.predict_proba(batch)[:, 1]
        for s, mid, score in zip(searches, mids, scores):
            s.update(mid, float(score), current_tier)

    suggestions: list[Counterfactual] = []
    for s in searches:
        if s.best is None:
            continue
        value, score, tier = s.best
        if s.kind == "single":
            current_value = float(X_base.iloc[0][s.name])
            suggestions.append(
                Counterfactual(
                    feature=s.name,
                    current_value=round(current_value, 4),
                    suggested_value=round(value, 4),
                    new_risk_score=round(score, 5),
                    new_tier=tier,
                    description=_describe(s.name, current_value, value, tier),
                )
            )
        else:
            suggestions.append(
                Counterfactual(
                    feature="combined",
                    current_value=0.0,
                    suggested_value=round(value, 4),
                    new_risk_score=round(score, 5),
                    new_tier=tier,
                    description=_describe_combined(value, tier),
                )
            )

    def relative_change(cf: Counterfactual) -> float:
        if cf.feature == "combined":
            return cf.suggested_value
        if cf.current_value == 0:
            return 1.0
        return abs(cf.current_value - cf.suggested_value) / cf.current_value

    suggestions.sort(key=relative_change)
    return suggestions[:max_suggestions]
