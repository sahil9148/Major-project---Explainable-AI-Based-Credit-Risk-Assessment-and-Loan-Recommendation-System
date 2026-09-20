"""
narrative.py
-------------
Turns a risk assessment + SHAP contributions into a plain-English
paragraph.

Always works via a template (no external dependency, zero cost, zero
latency risk). Optionally upgrades to a Claude-generated version if
ANTHROPIC_API_KEY is set in the environment, with a hard fallback to the
template if the API call fails for *any* reason -- an LLM outage, a
missing key, a network blip -- should never be able to break the
/predict endpoint. That's why llm_narrative() catches broadly instead of
letting exceptions propagate.
"""

from __future__ import annotations

import os

_RECOMMENDATION_TEXT = {
    "APPROVE": "the model recommends straightforward approval.",
    "APPROVE_WITH_CONDITIONS": "the model recommends approval with adjusted terms.",
    "MANUAL_REVIEW": (
        "the model recommends this go to manual underwriter review rather "
        "than an automated decision."
    ),
    "DENY": "the model recommends declining this application.",
}


def template_narrative(
    risk_score: float,
    risk_tier: str,
    recommendation: str,
    top_contributions: list,
) -> str:
    increasing = [c for c in top_contributions if c.direction == "increases_risk"][:2]
    decreasing = [c for c in top_contributions if c.direction == "decreases_risk"][:2]

    parts = [
        f"This application scored {risk_score * 100:.1f}% predicted default risk, "
        f"placing it in the {risk_tier.replace('_', ' ').title()} risk tier."
    ]
    if increasing:
        names = " and ".join(c.feature.replace("_", " ") for c in increasing)
        parts.append(f"The biggest factors pushing risk up were {names}.")
    if decreasing:
        names = " and ".join(c.feature.replace("_", " ") for c in decreasing)
        parts.append(f"Working in the applicant's favor: {names}.")

    rec_text = _RECOMMENDATION_TEXT.get(recommendation, "the model has made a recommendation.")
    parts.append(f"Given this, {rec_text}")
    return " ".join(parts)


def llm_narrative(
    risk_score: float,
    risk_tier: str,
    recommendation: str,
    top_contributions: list,
) -> "str | None":
    """Returns a Claude-generated narrative, or None if no API key is
    configured or the call fails for any reason -- caller falls back to
    template_narrative() in that case."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    try:
        from anthropic import Anthropic

        client = Anthropic(api_key=api_key)

        contributions_text = "\n".join(
            f"- {c.feature.replace('_', ' ')}: {c.direction.replace('_', ' ')} "
            f"(SHAP {c.shap_value:+.3f}, value={c.value})"
            for c in top_contributions[:6]
        )
        prompt = (
            "You are explaining an automated credit risk decision to a loan "
            "applicant in plain, respectful, non-technical language. Do not use "
            "the words 'SHAP' or 'model' more than once combined. Write 2-3 "
            "sentences, no headers, no bullet points, no markdown.\n\n"
            f"Predicted default risk: {risk_score * 100:.1f}%\n"
            f"Risk tier: {risk_tier}\n"
            f"Recommendation: {recommendation}\n"
            f"Top contributing factors:\n{contributions_text}"
        )

        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=250,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in message.content if block.type == "text").strip()
        return text or None
    except Exception:
        return None


def generate_narrative(
    risk_score: float,
    risk_tier: str,
    recommendation: str,
    top_contributions: list,
) -> str:
    llm_text = llm_narrative(risk_score, risk_tier, recommendation, top_contributions)
    if llm_text:
        return llm_text
    return template_narrative(risk_score, risk_tier, recommendation, top_contributions)
