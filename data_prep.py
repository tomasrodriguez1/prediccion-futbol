"""
data_prep.py

Data preparation module for the World Cup prediction system.
Handles:
  1. Filtering to men's World Cup matches only.
  2. Normalising historical team names to modern successors.
  3. Temporal decay weighting (exponential half-life).
  4. Leak-free "as-of" feature engineering (expanding weighted means
     computed STRICTLY from prior matches only).
"""

import os

import numpy as np
import pandas as pd

from elo import compute_elo_history, INTL_TEAM_NAME_MAP

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

MATCHES_CSV = "data-worldcup/data-csv/matches.csv"
INTL_MATCHES_CSV = "International Football Results/all_matches.csv"
EXTRA_MATCHES_CSV = "data-worldcup/extra_matches.csv"

# Half-life in years for the temporal decay weight.
# A match that is HALF_LIFE years old gets weight 0.5.
DEFAULT_HALF_LIFE: float = 8.0

# Recency boost: matches within RECENT_WINDOW_DAYS before the reference date get
# their decay weight multiplied by RECENT_BOOST. The 8-year half-life ALREADY
# makes recent matches weigh more (a 1-year-old match weighs ~0.92, a 16-year-old
# one ~0.25). An EXTRA boost on top was tested in backtest.py ("python backtest.py
# weights") across WC 1998-2022 and consistently WORSENED RPS/log-loss/Brier/
# accuracy, so it is disabled by default. The machinery is kept and fully
# parametrizable; set RECENT_BOOST > 1.0 to re-enable for experiments.
DEFAULT_RECENT_BOOST: float = 1.0
DEFAULT_RECENT_WINDOW_DAYS: int = 365

# Neutral prior for the "points per game" form feature (3=win, 1=draw, 0=loss).
NEUTRAL_PRIOR_POINTS: float = 1.2

# Map dissolved / renamed teams to their modern FIFA successor.
# Easy to extend: just add entries.
TEAM_NAME_MAP: dict[str, str] = {
    "West Germany":          "Germany",
    "East Germany":          "Germany",
    "Soviet Union":          "Russia",
    "USSR":                  "Russia",
    "Yugoslavia":            "Serbia",
    "Serbia and Montenegro": "Serbia",
    "Czechoslovakia":        "Czech Republic",
    "Dutch East Indies":     "Indonesia",
    "Zaire":                 "DR Congo",
    "United States":         "USA",
}

# Default prior when a team has no prior history at a given point in time.
# Will be overridden dynamically with the global weighted mean up to that date.
NEUTRAL_PRIOR_SCORED:   float = 1.35
NEUTRAL_PRIOR_CONCEDED: float = 1.35

# ---------------------------------------------------------------------------
# CONFEDERATION DATA
# ---------------------------------------------------------------------------

CONFEDERATION_MAP: dict[str, str] = {
    # UEFA
    "Spain": "UEFA", "France": "UEFA", "Germany": "UEFA", "England": "UEFA",
    "Portugal": "UEFA", "Netherlands": "UEFA", "Belgium": "UEFA", "Italy": "UEFA",
    "Croatia": "UEFA", "Denmark": "UEFA", "Switzerland": "UEFA", "Serbia": "UEFA",
    "Poland": "UEFA", "Austria": "UEFA", "Turkey": "UEFA", "Sweden": "UEFA",
    "Norway": "UEFA", "Hungary": "UEFA", "Romania": "UEFA", "Bulgaria": "UEFA",
    "Czech Republic": "UEFA", "Slovakia": "UEFA", "Slovenia": "UEFA",
    "Bosnia and Herzegovina": "UEFA", "Ukraine": "UEFA", "Scotland": "UEFA",
    "Wales": "UEFA", "Republic of Ireland": "UEFA", "Northern Ireland": "UEFA",
    "Greece": "UEFA", "Russia": "UEFA", "Albania": "UEFA", "Georgia": "UEFA",
    # CONMEBOL
    "Brazil": "CONMEBOL", "Argentina": "CONMEBOL", "Uruguay": "CONMEBOL",
    "Colombia": "CONMEBOL", "Ecuador": "CONMEBOL", "Chile": "CONMEBOL",
    "Peru": "CONMEBOL", "Paraguay": "CONMEBOL", "Bolivia": "CONMEBOL",
    "Venezuela": "CONMEBOL",
    # CONCACAF
    "USA": "CONCACAF", "Mexico": "CONCACAF", "Canada": "CONCACAF",
    "Panama": "CONCACAF", "Jamaica": "CONCACAF", "Honduras": "CONCACAF",
    "Costa Rica": "CONCACAF", "El Salvador": "CONCACAF", "Haiti": "CONCACAF",
    "Trinidad and Tobago": "CONCACAF", "Cuba": "CONCACAF", "Curaçao": "CONCACAF",
    # CAF
    "Senegal": "CAF", "Morocco": "CAF", "Nigeria": "CAF", "Cameroon": "CAF",
    "DR Congo": "CAF", "Ghana": "CAF", "Ivory Coast": "CAF", "Tunisia": "CAF",
    "Algeria": "CAF", "Egypt": "CAF", "South Africa": "CAF", "Angola": "CAF",
    "Togo": "CAF", "Zambia": "CAF", "Mali": "CAF", "Burkina Faso": "CAF",
    "Zaire": "CAF", "Cape Verde": "CAF",
    # AFC
    "Japan": "AFC", "South Korea": "AFC", "Iran": "AFC", "Saudi Arabia": "AFC",
    "Australia": "AFC", "Qatar": "AFC", "China PR": "AFC", "Iraq": "AFC",
    "North Korea": "AFC", "Indonesia": "AFC", "United Arab Emirates": "AFC",
    "Kuwait": "AFC", "Uzbekistan": "AFC", "Jordan": "AFC",
    # OFC
    "New Zealand": "OFC",
}

# Historical WC win rates per confederation (computed from 1930-2022 data)
CONFEDERATION_WIN_RATES: dict[str, float] = {
    "UEFA":     0.294,
    "CONMEBOL": 0.269,
    "CONCACAF": 0.120,
    "CAF":      0.119,
    "AFC":      0.102,
    "OFC":      0.050,
}


# ---------------------------------------------------------------------------
# TEMPORAL WEIGHTING
# ---------------------------------------------------------------------------

def compute_temporal_weight(
    match_dates: "pd.Series",
    ref_date=None,
    half_life: float = DEFAULT_HALF_LIFE,
    recent_boost: float = DEFAULT_RECENT_BOOST,
    recent_window_days: int = DEFAULT_RECENT_WINDOW_DAYS,
) -> "pd.Series":
    """Return exponential-decay weights with an optional recency boost.

    weight = exp(-ln2 * age_years / half_life), then multiplied by
    ``recent_boost`` for matches within ``recent_window_days`` before
    ``ref_date`` (defaults to the most recent match).

    Set ``recent_boost = 1.0`` to recover the plain half-life behaviour.
    """
    if ref_date is None:
        ref_date = match_dates.max()
    age_days = (ref_date - match_dates).dt.days
    age_years = age_days / 365.25
    weight = np.exp(-np.log(2) * age_years / half_life)
    if recent_boost != 1.0:
        is_recent = (age_days >= 0) & (age_days <= recent_window_days)
        weight = weight.where(~is_recent, weight * recent_boost)
    return weight


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def load_clean_matches(
    path: str = MATCHES_CSV,
    half_life: float = DEFAULT_HALF_LIFE,
) -> pd.DataFrame:
    """Return a clean DataFrame of men's World Cup matches.

    Steps applied (all in-memory, original CSV untouched):
      1. Drop Women's World Cup rows.
      2. Drop rows where score is null.
      3. Normalise team names via ``TEAM_NAME_MAP``.
      4. Add exponential decay ``weight`` column.
      5. Sort ascending by ``match_date``.

    Parameters
    ----------
    path : str
        Path to ``matches.csv``.
    half_life : float
        Half-life (years) for the temporal-decay weight.

    Returns
    -------
    pd.DataFrame
        Cleaned frame, sorted by date, with column ``weight`` added.
    """
    df = pd.read_csv(path)

    # 1. Keep only men's World Cup
    df = df[~df["tournament_name"].str.contains("Women", case=False, na=False)].copy()

    # 2. Drop null scores
    df = df.dropna(subset=["home_team_score", "away_team_score"])

    # Ensure scores are integers
    df["home_team_score"] = df["home_team_score"].astype(int)
    df["away_team_score"] = df["away_team_score"].astype(int)

    # 3. Normalise team names
    df["home_team_name"] = df["home_team_name"].replace(TEAM_NAME_MAP)
    df["away_team_name"] = df["away_team_name"].replace(TEAM_NAME_MAP)

    # 4. Parse date and compute temporal weight
    df["match_date"] = pd.to_datetime(df["match_date"], format="%Y-%m-%d")
    df["weight"] = compute_temporal_weight(df["match_date"], half_life=half_life)

    # 5. Sort by date ascending (critical for temporal features)
    df = df.sort_values("match_date").reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# INTERNATIONAL MATCHES (non-World-Cup) + COMBINED CORPUS
# ---------------------------------------------------------------------------

# Columns shared by the World Cup and international match frames once
# normalised. This is the schema the rest of the pipeline operates on.
_CORPUS_COLUMNS = [
    "match_date", "home_team_name", "away_team_name",
    "home_team_score", "away_team_score", "country_name",
    "tournament_id", "is_world_cup",
]


def load_international_matches(path: str = INTL_MATCHES_CSV) -> pd.DataFrame:
    """Load the international results dataset, normalised to the standard schema.

    World Cup final-tournament matches are dropped (they are duplicated by
    ``load_clean_matches``, which is the canonical source for those).

    Returns
    -------
    pd.DataFrame
        Columns: ``match_date, home_team_name, away_team_name,
        home_team_score, away_team_score, country_name, tournament_id,
        is_world_cup`` (always 0 here).
    """
    df = pd.read_csv(path)

    # Drop World Cup final-tournament matches — already covered by matches.csv
    df = df[df["tournament"] != "World Cup"].copy()

    # Normalise team names (dataset-specific aliases, then historical→modern)
    df["home_team"] = df["home_team"].replace(INTL_TEAM_NAME_MAP).replace(TEAM_NAME_MAP)
    df["away_team"] = df["away_team"].replace(INTL_TEAM_NAME_MAP).replace(TEAM_NAME_MAP)

    df = df.rename(columns={
        "home_team": "home_team_name",
        "away_team": "away_team_name",
        "home_score": "home_team_score",
        "away_score": "away_team_score",
        "country": "country_name",
        "tournament": "tournament_id",
    })

    df["match_date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df["is_world_cup"] = 0

    return df[_CORPUS_COLUMNS]


def load_extra_matches(path: str = EXTRA_MATCHES_CSV) -> pd.DataFrame:
    """Load user-added matches (manual entries + API-synced results).

    Same raw schema as ``all_matches.csv``:
        date, home_team, away_team, home_score, away_score, tournament, country

    Names are normalised with the inference alias map (e.g. "Czechia" →
    "Czech Republic") plus the historical → modern map. ``is_world_cup`` is 1
    when ``tournament == "World Cup"``.

    Returns an empty, correctly-typed frame if the file does not exist.
    """
    if not os.path.exists(path):
        empty = pd.DataFrame(columns=_CORPUS_COLUMNS)
        empty["match_date"] = pd.to_datetime(empty["match_date"])
        return empty

    df = pd.read_csv(path)
    if df.empty:
        empty = pd.DataFrame(columns=_CORPUS_COLUMNS)
        empty["match_date"] = pd.to_datetime(empty["match_date"])
        return empty

    # Inference aliases (API/schedule) first, then historical → modern.
    from team_names import INFERENCE_TEAM_NAME_MAP
    df["home_team"] = (df["home_team"].replace(INFERENCE_TEAM_NAME_MAP)
                       .replace(INTL_TEAM_NAME_MAP).replace(TEAM_NAME_MAP))
    df["away_team"] = (df["away_team"].replace(INFERENCE_TEAM_NAME_MAP)
                       .replace(INTL_TEAM_NAME_MAP).replace(TEAM_NAME_MAP))

    df = df.rename(columns={
        "home_team": "home_team_name",
        "away_team": "away_team_name",
        "home_score": "home_team_score",
        "away_score": "away_team_score",
        "country": "country_name",
        "tournament": "tournament_id",
    })

    df["match_date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df["is_world_cup"] = (df["tournament_id"] == "World Cup").astype(int)

    return df[_CORPUS_COLUMNS]


def load_combined_corpus(
    half_life: float = DEFAULT_HALF_LIFE,
    recent_boost: float = DEFAULT_RECENT_BOOST,
    recent_window_days: int = DEFAULT_RECENT_WINDOW_DAYS,
) -> pd.DataFrame:
    """Return the World Cup + international + user-added matches as one corpus.

    World Cup matches are tagged ``is_world_cup = 1``; all other international
    matches (qualifiers, friendlies, continental cups, ...) are tagged
    ``is_world_cup = 0``. User-added matches (``extra_matches.csv``) are appended
    and deduplicated against the historical data by
    ``(match_date, home_team_name, away_team_name)``.

    The temporal-decay ``weight`` is recomputed over the combined date range
    with the configured recency boost.

    Returns
    -------
    pd.DataFrame
        Sorted ascending by ``match_date``, with columns from
        ``_CORPUS_COLUMNS`` plus ``weight``.
    """
    wc = load_clean_matches(half_life=half_life)
    wc = wc.copy()
    wc["is_world_cup"] = 1
    wc = wc[_CORPUS_COLUMNS]

    intl = load_international_matches()
    extra = load_extra_matches()

    # Only concat non-empty frames: an all-object empty frame would upcast the
    # numeric score columns to ``object`` and break downstream cumsum/weighting.
    frames = [f for f in (wc, intl, extra) if not f.empty]
    combined = pd.concat(frames, ignore_index=True)

    # Defensive dtype enforcement (user-added rows may arrive as strings).
    combined["home_team_score"] = combined["home_team_score"].astype(int)
    combined["away_team_score"] = combined["away_team_score"].astype(int)

    # Deduplicate: keep the first occurrence per (date, home, away). Historical
    # rows come first, so a user-added duplicate of an existing match is dropped.
    combined = combined.drop_duplicates(
        subset=["match_date", "home_team_name", "away_team_name"],
        keep="first",
    )

    combined = combined.sort_values("match_date").reset_index(drop=True)

    # Recompute the temporal-decay weight (with recency boost) over the range.
    combined["weight"] = compute_temporal_weight(
        combined["match_date"],
        half_life=half_life,
        recent_boost=recent_boost,
        recent_window_days=recent_window_days,
    )

    return combined


# ---------------------------------------------------------------------------
# HOST-NATION FLAGS
# ---------------------------------------------------------------------------

def add_host_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add host-nation flags to each match row.

    A team is the host if it plays in its own country.  We normalise
    ``country_name`` with the same ``TEAM_NAME_MAP`` used for team names
    so that e.g. "United States" matches "USA".

    New columns added
    -----------------
    host_home : int (0 or 1)
        1 if the home team is the host nation.
    host_away : int (0 or 1)
        1 if the away team is the host nation.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with ``host_home`` and ``host_away`` added.
    """
    df = df.copy()
    country_norm = df["country_name"].replace(TEAM_NAME_MAP)
    df["host_home"] = (df["home_team_name"] == country_norm).astype(int)
    df["host_away"] = (df["away_team_name"] == country_norm).astype(int)
    return df


# ---------------------------------------------------------------------------
# CONFEDERATION FEATURES
# ---------------------------------------------------------------------------

def add_confederation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add historical WC win-rate per confederation for home and away teams.

    New columns
    -----------
    conf_strength_home : float  (e.g. 0.294 for UEFA, 0.102 for AFC)
    conf_strength_away : float
    """
    df = df.copy()

    def _conf_strength(team: str) -> float:
        conf = CONFEDERATION_MAP.get(team, "UEFA")
        return CONFEDERATION_WIN_RATES.get(conf, 0.294)

    df["conf_strength_home"] = df["home_team_name"].apply(_conf_strength)
    df["conf_strength_away"] = df["away_team_name"].apply(_conf_strength)
    return df


# ---------------------------------------------------------------------------
# LEAK-FREE FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def unify_team_history(df: pd.DataFrame) -> pd.DataFrame:
    """Stack home and away records into a single chronological series per team.

    Returns a long-form DataFrame with columns:
        team, match_date, goals_scored, goals_conceded, weight, match_idx
    sorted by (team, match_date, match_idx).
    """
    home = df[["match_date", "home_team_name", "home_team_score",
               "away_team_score", "weight"]].copy()
    home.columns = ["match_date", "team", "goals_scored", "goals_conceded", "weight"]
    home["match_idx"] = df.index

    away = df[["match_date", "away_team_name", "away_team_score",
               "home_team_score", "weight"]].copy()
    away.columns = ["match_date", "team", "goals_scored", "goals_conceded", "weight"]
    away["match_idx"] = df.index

    history = pd.concat([home, away], ignore_index=True)
    history = history.sort_values(["team", "match_date", "match_idx"]).reset_index(drop=True)
    return history


def build_asof_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach leak-free as-of features to every row of *df*.

    For each match, the features for a team are the weighted expanding mean
    of goals scored / conceded using ONLY matches STRICTLY before that match.
    This is achieved via cumulative sums + ``shift(1)`` so that the current
    row is never included.

    When a team has no prior history, the global weighted average up to that
    date is used as a neutral prior.

    New columns added
    -----------------
    h_avg_scored, h_avg_conceded : float
        Home team's prior weighted averages.
    a_avg_scored, a_avg_conceded : float
        Away team's prior weighted averages.
    elo_home, elo_away, elo_diff : float
        Pre-match Elo ratings and their difference (leak-free).
    host_home, host_away : int
        1 if the team is the actual host nation, 0 otherwise.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with the nine feature columns.
    """
    df = df.copy()

    # -- 0a. Host flags (needed before Elo for HA calculation) -------------
    df = add_host_flags(df)

    # -- 0b. Elo ratings (sequential, leak-free by construction) -----------
    df = compute_elo_history(df)

    # -- 1. Build unified per-team history ---------------------------------
    history = unify_team_history(df)

    # -- 2. Weighted expanding mean per team (shifted to exclude current) --
    # cumulative weighted sum BEFORE current match = shift(1) of cumsum
    history["w_scored"]   = history["goals_scored"]   * history["weight"]
    history["w_conceded"] = history["goals_conceded"] * history["weight"]

    # Group by team and compute shifted cumulative sums.
    # shift(1) ensures the current match is EXCLUDED.  ← LEAK-FREE GUARANTEE
    grp = history.groupby("team")
    history["cum_w"]          = grp["weight"].cumsum().shift(1)
    history["cum_w_scored"]   = grp["w_scored"].cumsum().shift(1)
    history["cum_w_conceded"] = grp["w_conceded"].cumsum().shift(1)

    # Per-row shift only applies within each team group; fill NaN for first
    # matches of each team (done by the shift already producing NaN).

    history["avg_scored"]   = history["cum_w_scored"]   / history["cum_w"]
    history["avg_conceded"] = history["cum_w_conceded"] / history["cum_w"]

    # -- 2b. Recent-form features: rolling mean of the last 5 matches --------
    # (goals scored, goals conceded, points: win=3 / draw=1 / loss=0)
    # shift(1) excludes the current match  ← LEAK-FREE GUARANTEE
    history["points"] = np.select(
        [history["goals_scored"] > history["goals_conceded"],
         history["goals_scored"] == history["goals_conceded"]],
        [3.0, 1.0],
        default=0.0,
    )

    grp = history.groupby("team")
    history["form_scored"]   = grp["goals_scored"].transform(
        lambda s: s.rolling(5, min_periods=1).mean().shift(1))
    history["form_conceded"] = grp["goals_conceded"].transform(
        lambda s: s.rolling(5, min_periods=1).mean().shift(1))
    history["form_points"]   = grp["points"].transform(
        lambda s: s.rolling(5, min_periods=1).mean().shift(1))

    # -- 3. Global weighted prior up to each date (for teams with no history)
    # We compute a single global expanding weighted mean over the full df.
    global_w_scored   = (df["home_team_score"] * df["weight"]).cumsum() + \
                        (df["away_team_score"] * df["weight"]).cumsum()
    global_w_conceded = global_w_scored  # symmetric: every goal scored is conceded
    global_cum_w      = df["weight"].cumsum() * 2  # two team-observations per match
    global_avg = (global_w_scored / global_cum_w).shift(1)
    global_avg.iloc[0] = NEUTRAL_PRIOR_SCORED  # very first match: use static prior

    # Map global prior back by match_idx
    global_prior_map = dict(zip(df.index, global_avg))

    # Fill NaN averages (first appearance of a team) with the global prior.
    history["global_prior"] = history["match_idx"].map(global_prior_map)
    history["avg_scored"]   = history["avg_scored"].fillna(history["global_prior"])
    history["avg_conceded"] = history["avg_conceded"].fillna(history["global_prior"])

    # Form features: a team's very first-ever match has no prior games to
    # roll over (NaN). Fall back to the same global prior for goals, and a
    # static neutral prior for points.
    history["form_scored"]   = history["form_scored"].fillna(history["global_prior"])
    history["form_conceded"] = history["form_conceded"].fillna(history["global_prior"])
    history["form_points"]   = history["form_points"].fillna(NEUTRAL_PRIOR_POINTS)

    # -- 4. Map back to original df rows -----------------------------------
    # For each match_idx we need home-team stats and away-team stats.
    stat_cols = ["avg_scored", "avg_conceded", "form_scored", "form_conceded", "form_points"]

    home_stats = (
        history[history["match_idx"].isin(df.index)]
        .merge(
            df[["home_team_name"]].rename_axis("match_idx"),
            left_on=["match_idx", "team"],
            right_on=["match_idx", "home_team_name"],
            how="inner",
        )
        .set_index("match_idx")[stat_cols]
        .rename(columns={
            "avg_scored": "h_avg_scored", "avg_conceded": "h_avg_conceded",
            "form_scored": "h_form_scored", "form_conceded": "h_form_conceded",
            "form_points": "h_form_points",
        })
    )

    away_stats = (
        history[history["match_idx"].isin(df.index)]
        .merge(
            df[["away_team_name"]].rename_axis("match_idx"),
            left_on=["match_idx", "team"],
            right_on=["match_idx", "away_team_name"],
            how="inner",
        )
        .set_index("match_idx")[stat_cols]
        .rename(columns={
            "avg_scored": "a_avg_scored", "avg_conceded": "a_avg_conceded",
            "form_scored": "a_form_scored", "form_conceded": "a_form_conceded",
            "form_points": "a_form_points",
        })
    )

    df = df.join(home_stats).join(away_stats)

    # Safety: any remaining NaN gets the static neutral prior
    for col in ["h_avg_scored", "h_avg_conceded", "a_avg_scored", "a_avg_conceded",
                "h_form_scored", "a_form_scored", "h_form_conceded", "a_form_conceded"]:
        df[col] = df[col].fillna(NEUTRAL_PRIOR_SCORED)
    for col in ["h_form_points", "a_form_points"]:
        df[col] = df[col].fillna(NEUTRAL_PRIOR_POINTS)

    return df


def add_squad_values(df: pd.DataFrame, path: str = "data-worldcup/squad_values.csv") -> pd.DataFrame:
    """Add squad market values as features for both teams.

    ``squad_values.csv`` only has one entry per team per World Cup year. For
    matches in other years (e.g. qualifiers, friendlies), each team's value
    is taken from its NEAREST available World Cup year via an as-of merge.
    Teams with no squad-value history at all fall back to the global median.
    """
    if not os.path.exists(path):
        import logging
        logging.warning(f"{path} no existe. Retornando dataframe sin squad values.")
        return df

    df = df.copy()
    # Extraer año del torneo
    df["year"] = df["match_date"].dt.year

    sv_df = pd.read_csv(path)
    global_median = sv_df["squad_value_eur"].median()

    def _nearest_value(team_col: str) -> np.ndarray:
        # Use a fresh positional index (`_row`) since `df` may have a
        # non-unique index after upstream joins.
        left = pd.DataFrame({
            "_row": np.arange(len(df)),
            "team_name": df[team_col].to_numpy(),
            "year": df["year"].astype("int64").to_numpy(),
        }).sort_values("year")
        right = sv_df[["team_name", "year", "squad_value_eur"]].copy()
        right["year"] = right["year"].astype("int64")
        right = right.sort_values("year")
        merged = pd.merge_asof(left, right, on="year", by="team_name", direction="nearest")
        return merged.sort_values("_row")["squad_value_eur"].to_numpy()

    df["sv_home_eur"] = pd.Series(_nearest_value("home_team_name")).fillna(global_median).to_numpy()
    df["sv_away_eur"] = pd.Series(_nearest_value("away_team_name")).fillna(global_median).to_numpy()

    # Escalar a millones de EUR
    df["sv_home"] = df["sv_home_eur"] / 1e6
    df["sv_away"] = df["sv_away_eur"] / 1e6
    
    # Calcular ratio
    df["sv_ratio"] = df["sv_home"] / df["sv_away"].clip(lower=1.0)
    
    # Cleanup
    df = df.drop(columns=["year", "sv_home_eur", "sv_away_eur"], errors="ignore")
    return df

# ---------------------------------------------------------------------------
# Quick sanity check when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    clean = load_clean_matches()
    print(f"Clean matches: {len(clean)} rows")
    print(f"Date range:    {clean['match_date'].min().date()} → {clean['match_date'].max().date()}")
    print(f"Teams:         {clean['home_team_name'].nunique() + clean['away_team_name'].nunique()} (non-unique count)")
    print(f"Weight range:  {clean['weight'].min():.4f} → {clean['weight'].max():.4f}")
    print(f"\nSample weights (first 3 + last 3):")
    print(clean[["match_date", "home_team_name", "away_team_name", "weight"]].head(3).to_string(index=False))
    print("...")
    print(clean[["match_date", "home_team_name", "away_team_name", "weight"]].tail(3).to_string(index=False))

    featured = build_asof_features(clean)
    print(f"\nFeature columns added. Sample (rows 5-10):")
    print(featured[["home_team_name", "away_team_name",
                     "h_avg_scored", "h_avg_conceded",
                     "a_avg_scored", "a_avg_conceded"]].iloc[5:11].to_string(index=False))
    # Verify no NaN in features
    nan_count = featured[["h_avg_scored", "h_avg_conceded", "a_avg_scored", "a_avg_conceded"]].isna().sum().sum()
    print(f"\nNaN in features: {nan_count}")
