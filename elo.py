"""
elo.py

Dynamic Elo rating system for national football teams.

Computes a sequential Elo rating that evolves match-by-match.  For each match,
the PRE-match Elo is recorded as a feature BEFORE the rating is updated with
the result.  This guarantees the Elo features are leak-free by construction.

Usage
-----
    from elo import compute_elo_history
    df = compute_elo_history(df, k=40, home_adv_elo=100, mov_factor=True)
    # df now has columns: elo_home, elo_away, elo_diff
"""

import numpy as np
import pandas as pd


# Default starting Elo for teams with no prior history.
DEFAULT_ELO: float = 1500.0


def _margin_of_victory_multiplier(goal_diff: int) -> float:
    """Return the margin-of-victory multiplier for Elo updates.

    Standard football Elo scaling:
      - 1-goal difference → 1.0
      - 2-goal difference → 1.5
      - 3+ goal difference → (11 + diff) / 8

    Parameters
    ----------
    goal_diff : int
        Absolute goal difference (always >= 0).

    Returns
    -------
    float
        Multiplier >= 1.0.
    """
    diff = abs(goal_diff)
    if diff <= 1:
        return 1.0
    elif diff == 2:
        return 1.5
    else:
        return (11 + diff) / 8


def compute_elo_history(
    df: pd.DataFrame,
    k: float = 40.0,
    home_adv_elo: float = 100.0,
    mov_factor: bool = True,
) -> pd.DataFrame:
    """Add Elo-based features to every row of *df*.

    The DataFrame must already contain columns:
      - ``home_team_name``, ``away_team_name``
      - ``home_team_score``, ``away_team_score``
      - ``host_home`` (1 if home team is the host nation, else 0)

    The DataFrame **must be sorted by match_date ascending** (the standard
    output of ``data_prep.load_clean_matches``).

    New columns added
    -----------------
    elo_home : float
        Home team's Elo rating BEFORE this match.
    elo_away : float
        Away team's Elo rating BEFORE this match.
    elo_diff : float
        elo_home − elo_away (positive means home team is stronger).

    Parameters
    ----------
    df : pd.DataFrame
        Match data, sorted chronologically.
    k : float
        K-factor controlling how quickly Elo responds to results.
        Default 40 (standard for international football).
    home_adv_elo : float
        Elo points added to the expected score calculation when the home
        team is the actual host nation (``host_home == 1``).
        Default 100 (≈ 0.64 expected score for host).
    mov_factor : bool
        Whether to apply the margin-of-victory multiplier.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with ``elo_home``, ``elo_away``, ``elo_diff`` added.
    """
    df = df.copy()

    # Elo ratings dictionary: {team_name: current_elo}
    elo_ratings: dict[str, float] = {}

    # Pre-allocate arrays for the new columns
    n = len(df)
    elo_home_arr = np.empty(n, dtype=np.float64)
    elo_away_arr = np.empty(n, dtype=np.float64)

    for i in range(n):
        row = df.iloc[i]
        home_team = row["home_team_name"]
        away_team = row["away_team_name"]
        home_score = int(row["home_team_score"])
        away_score = int(row["away_team_score"])

        # --- STEP 1: Record PRE-match Elo (BEFORE update) ← LEAK-FREE ---
        # These are the feature values.  They depend only on PRIOR matches,
        # never on the current match's result.
        elo_h = elo_ratings.get(home_team, DEFAULT_ELO)
        elo_a = elo_ratings.get(away_team, DEFAULT_ELO)
        elo_home_arr[i] = elo_h
        elo_away_arr[i] = elo_a

        # --- STEP 2: Compute expected scores --------------------------------
        # Host advantage: only the REAL host nation gets a home-field bonus.
        # In World Cup matches, most games are neutral → HA = 0.
        is_host_home = int(row.get("host_home", 0))
        ha = home_adv_elo * is_host_home

        expected_home = 1.0 / (1.0 + 10.0 ** (-(elo_h + ha - elo_a) / 400.0))
        expected_away = 1.0 - expected_home

        # --- STEP 3: Actual result ------------------------------------------
        if home_score > away_score:
            actual_home = 1.0
        elif home_score == away_score:
            actual_home = 0.5
        else:
            actual_home = 0.0
        actual_away = 1.0 - actual_home

        # --- STEP 4: Margin-of-victory multiplier ---------------------------
        if mov_factor:
            mov = _margin_of_victory_multiplier(home_score - away_score)
        else:
            mov = 1.0

        # --- STEP 5: Update Elo (AFTER recording pre-match values) ----------
        elo_ratings[home_team] = elo_h + k * mov * (actual_home - expected_home)
        elo_ratings[away_team] = elo_a + k * mov * (actual_away - expected_away)

    # Assign columns
    df["elo_home"] = elo_home_arr
    df["elo_away"] = elo_away_arr
    df["elo_diff"] = elo_home_arr - elo_away_arr

    return df


def get_final_elo_ratings(df: pd.DataFrame, **kwargs) -> dict[str, float]:
    """Compute Elo history and return the final rating per team.

    This is used by ``train_model.py`` to export Elo ratings for inference.

    Parameters
    ----------
    df : pd.DataFrame
        Full match history (with ``host_home`` column), sorted by date.
    **kwargs
        Forwarded to ``compute_elo_history`` (k, home_adv_elo, mov_factor).

    Returns
    -------
    dict[str, float]
        {team_name: final_elo_rating} after processing all matches.
    """
    # We need to replay the full Elo history to get final ratings
    elo_ratings: dict[str, float] = {}

    k = kwargs.get("k", 40.0)
    home_adv_elo = kwargs.get("home_adv_elo", 100.0)
    mov_factor = kwargs.get("mov_factor", True)

    for i in range(len(df)):
        row = df.iloc[i]
        home_team = row["home_team_name"]
        away_team = row["away_team_name"]
        home_score = int(row["home_team_score"])
        away_score = int(row["away_team_score"])

        elo_h = elo_ratings.get(home_team, DEFAULT_ELO)
        elo_a = elo_ratings.get(away_team, DEFAULT_ELO)

        is_host_home = int(row.get("host_home", 0))
        ha = home_adv_elo * is_host_home

        expected_home = 1.0 / (1.0 + 10.0 ** (-(elo_h + ha - elo_a) / 400.0))

        if home_score > away_score:
            actual_home = 1.0
        elif home_score == away_score:
            actual_home = 0.5
        else:
            actual_home = 0.0

        if mov_factor:
            mov = _margin_of_victory_multiplier(home_score - away_score)
        else:
            mov = 1.0

        elo_ratings[home_team] = elo_h + k * mov * (actual_home - expected_home)
        elo_ratings[away_team] = elo_a + k * mov * ((1.0 - actual_home) - (1.0 - expected_home))

    return elo_ratings


# Name mapping: this dataset's country names → our standard TEAM_NAME_MAP names
INTL_TEAM_NAME_MAP: dict[str, str] = {
    "United States":            "USA",
    "Korea":                    "South Korea",
    "Comm of Indep States":     "Russia",
    "United Arab Republic":     "Egypt",
    "Czechia":                  "Czech Republic",
    "Ireland":                  "Republic of Ireland",
}


# ---------------------------------------------------------------------------
# Quick sanity check when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from data_prep import load_clean_matches, TEAM_NAME_MAP

    df = load_clean_matches()

    # Add host flags for the Elo computation
    df["country_norm"] = df["country_name"].replace(TEAM_NAME_MAP)
    df["host_home"] = (df["home_team_name"] == df["country_norm"]).astype(int)
    df["host_away"] = (df["away_team_name"] == df["country_norm"]).astype(int)

    df = compute_elo_history(df)

    print("Elo feature sample (last 10 matches):")
    print(df[["match_date", "home_team_name", "away_team_name",
              "elo_home", "elo_away", "elo_diff"]].tail(10).to_string(index=False))

    print(f"\nElo range: {df['elo_home'].min():.0f} – {df['elo_home'].max():.0f}")
    print(f"NaN count: {df[['elo_home', 'elo_away', 'elo_diff']].isna().sum().sum()}")

    # Final ratings
    finals = get_final_elo_ratings(df)
    top10 = sorted(finals.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 Elo ratings (after all WC matches):")
    for team, elo in top10:
        print(f"  {team:25s} {elo:.1f}")
