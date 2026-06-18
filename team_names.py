"""
team_names.py

Canonical team-name normalization for inference (predictor, simulation, app).
Maps API / schedule aliases to the names used in models/ and data_prep.
"""

import re

from elo import INTL_TEAM_NAME_MAP

# API, schedule, and football-data.org aliases → canonical model names.
INFERENCE_TEAM_NAME_MAP: dict[str, str] = {
    **INTL_TEAM_NAME_MAP,
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Côte d'Ivoire": "Ivory Coast",
    "Korea Republic": "South Korea",
    "Republic of Korea": "South Korea",
    "Korea DPR": "North Korea",
    "IR Iran": "Iran",
    # football-data.org usa estos nombres; el calendario/modelos usan los canónicos.
    "Congo DR": "DR Congo",
    "Congo": "DR Congo",
    "Democratic Republic of Congo": "DR Congo",
    "Cape Verde Islands": "Cape Verde",
    "Cabo Verde": "Cape Verde",
}


def normalize_team_name(name: str | None) -> str | None:
    """Return the canonical team name used by the ML models."""
    if not name:
        return name
    return INFERENCE_TEAM_NAME_MAP.get(name, name)


def is_placeholder_team(name: str | None) -> bool:
    """True for knockout placeholders (W74, 1A, 3A/B/C/D/F, TBD, ...).

    All real placeholders contain a digit or a slash; real team names never do.
    """
    if not name or name == "TBD":
        return True
    if re.search(r"\d", name):
        return True
    if "/" in name:
        return True
    return False
