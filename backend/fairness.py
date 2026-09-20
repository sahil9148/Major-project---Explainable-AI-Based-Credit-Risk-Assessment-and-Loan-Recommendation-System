"""
fairness.py
------------
Group fairness audit across age brackets. Age is both an existing model
feature and a class explicitly protected against discrimination in credit
decisions under the US Equal Credit Opportunity Act (ECOA) -- which makes
it a legitimate, realistic choice for this audit, rather than inventing a
synthetic demographic field that doesn't correspond to anything real.

Computed once at training time (not per-request) and served statically,
since it's a property of the model as a whole, not of any one applicant.

Metrics, per age group, on the held-out test set:
  - selection_rate: share the model would approve or conditionally approve
  - true_positive_rate: of people who did NOT actually default, the share
    correctly identified as low-enough risk to approve ("equal opportunity")
  - actual vs. predicted default rate: is the model's risk score equally
    meaningful across groups, or miscalibrated for some of them?

Then reports the four-fifths rule: a standard (though not sole) US
EEOC/disparate-impact screening threshold -- if any group's selection rate
is less than 80% of the highest group's, that's a standard trigger for
further fairness investigation, not an automatic finding of discrimination.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

AGE_BUCKETS = [
    (18, 25, "18-25"),
    (26, 40, "26-40"),
    (41, 60, "41-60"),
    (61, 200, "61+"),
]

# Matches build_recommendation()'s APPROVE / APPROVE_WITH_CONDITIONS cutoff
# in main.py -- "selected" means the model would approve in some form.
APPROVAL_SCORE_THRESHOLD = 0.40

MIN_GROUP_SIZE_FOR_RATIO = 20  # ignore tiny groups when computing the headline ratio


def age_to_bucket(age: float) -> str:
    for lo, hi, label in AGE_BUCKETS:
        if lo <= age <= hi:
            return label
    return AGE_BUCKETS[-1][2]


def compute_fairness_report(
    ages: np.ndarray, y_true: np.ndarray, y_proba: np.ndarray
) -> dict:
    df = pd.DataFrame({"age": ages, "y_true": y_true, "y_proba": y_proba})
    df["group"] = df["age"].apply(age_to_bucket)
    df["approved"] = df["y_proba"] < APPROVAL_SCORE_THRESHOLD

    groups = []
    for _, _, label in AGE_BUCKETS:
        g = df[df["group"] == label]
        if len(g) == 0:
            continue
        non_defaulters = g[g["y_true"] == 0]
        tpr = float(non_defaulters["approved"].mean()) if len(non_defaulters) > 0 else None
        groups.append(
            {
                "group": label,
                "n": int(len(g)),
                "selection_rate": round(float(g["approved"].mean()), 4),
                "actual_default_rate": round(float(g["y_true"].mean()), 4),
                "mean_predicted_risk": round(float(g["y_proba"].mean()), 4),
                "true_positive_rate": round(tpr, 4) if tpr is not None else None,
            }
        )

    eligible_rates = [g["selection_rate"] for g in groups if g["n"] >= MIN_GROUP_SIZE_FOR_RATIO]
    if len(eligible_rates) >= 2 and max(eligible_rates) > 0:
        four_fifths_ratio = round(min(eligible_rates) / max(eligible_rates), 4)
    else:
        four_fifths_ratio = None

    return {
        "groups": groups,
        "four_fifths_ratio": four_fifths_ratio,
        "four_fifths_pass": (four_fifths_ratio is not None and four_fifths_ratio >= 0.8),
        "methodology": (
            "Selection rate = share of each age group the model would approve or "
            "conditionally approve. The four-fifths rule (a standard US EEOC "
            "disparate-impact screening threshold) flags an issue when any group's "
            "selection rate falls below 80% of the highest group's rate. This is a "
            "screening heuristic used to prompt further investigation, not a legal "
            "determination of discrimination on its own."
        ),
    }
