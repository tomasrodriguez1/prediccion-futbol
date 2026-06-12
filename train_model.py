"""
train_model.py

Trains two Random Forest regressors (home goals, away goals) on historical
World Cup data, using the cleaned + leak-free pipeline from data_prep.py.

The trained models and per-team stats are exported to the ``models/`` directory
for use by ``predictor.py`` at inference time in the Streamlit app.

Usage
-----
    source .venv/bin/activate && python train_model.py
"""

import json
import os

import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor

from data_prep import (
    load_combined_corpus, build_asof_features, add_host_flags,
    add_squad_values, add_confederation_features, unify_team_history,
)
from elo import get_final_elo_ratings

# Matches before this date are used only to seed Elo ratings and recent-form
# history; they are not used as RF training targets (too old / different era).
MIN_TRAIN_DATE = "1990-01-01"

# Base feature columns used by the Random Forest models, in order.
# Squad-value and confederation columns are appended conditionally in
# prepare_data() depending on data availability.
BASE_FEATURE_COLS = [
    "h_avg_scored", "h_avg_conceded", "a_avg_scored", "a_avg_conceded",
    "h_form_scored", "h_form_conceded", "h_form_points",
    "a_form_scored", "a_form_conceded", "a_form_points",
    "elo_home", "elo_away", "elo_diff",
    "host_home", "host_away",
    "is_world_cup",
]


def _compute_current_form(df, n=5):
    """Per-team recent-form snapshot (last *n* matches) as of the end of *df*.

    Used for inference: a future World Cup match's form features are simply
    each team's form from its most recent matches in the combined corpus.
    """
    history = unify_team_history(df)
    history["points"] = np.select(
        [history["goals_scored"] > history["goals_conceded"],
         history["goals_scored"] == history["goals_conceded"]],
        [3.0, 1.0],
        default=0.0,
    )

    form = {}
    for team, group in history.groupby("team"):
        last_n = group.tail(n)
        form[team] = {
            "form_scored": round(float(last_n["goals_scored"].mean()), 4),
            "form_conceded": round(float(last_n["goals_conceded"].mean()), 4),
            "form_points": round(float(last_n["points"].mean()), 4),
        }
    return form


def prepare_data():
    """Load, clean, and build leak-free features for the full dataset.

    Returns
    -------
    X : np.ndarray
        Feature matrix, columns as in ``FEATURE_COLS``.
    y_home : np.ndarray
        Home team goals (target).
    y_away : np.ndarray
        Away team goals (target).
    sample_weights : np.ndarray
        Temporal decay weights for each match.
    team_stats : dict
        Per-team weighted averages (computed over ALL history — for inference only,
        not for training features).
    elo_ratings : dict
        Final Elo rating per team after the full combined corpus.
    team_form : dict
        Per-team recent-form snapshot (last 5 matches) for inference.
    """
    df = load_combined_corpus()
    featured = build_asof_features(df)
    featured = add_squad_values(featured)
    featured = add_confederation_features(featured)

    feature_cols = list(BASE_FEATURE_COLS)
    if "sv_home" in featured.columns:
        feature_cols.extend(["sv_home", "sv_away", "sv_ratio"])
    if "conf_strength_home" in featured.columns:
        feature_cols.extend(["conf_strength_home", "conf_strength_away"])

    # RF training rows: drop matches before MIN_TRAIN_DATE. Older matches
    # still contributed to the as-of features (Elo, averages, form) above.
    train_rows = featured[featured["match_date"] >= MIN_TRAIN_DATE]

    X = train_rows[feature_cols].values
    y_home = train_rows["home_team_score"].values
    y_away = train_rows["away_team_score"].values
    sample_weights = train_rows["weight"].values

    # --- Team stats for inference (Streamlit predictor) ---
    # At inference time, we want the FULL weighted average per team.
    # This is NOT used for training features (those are leak-free as-of).
    team_stats = {}
    teams = set(df["home_team_name"].unique()) | set(df["away_team_name"].unique())

    for team in teams:
        home_mask = df["home_team_name"] == team
        away_mask = df["away_team_name"] == team

        w_home = df.loc[home_mask, "weight"].values
        w_away = df.loc[away_mask, "weight"].values

        scored_home = df.loc[home_mask, "home_team_score"].values
        scored_away = df.loc[away_mask, "away_team_score"].values
        conceded_home = df.loc[home_mask, "away_team_score"].values
        conceded_away = df.loc[away_mask, "home_team_score"].values

        all_scored = np.concatenate([scored_home, scored_away])
        all_conceded = np.concatenate([conceded_home, conceded_away])
        all_weights = np.concatenate([w_home, w_away])

        total_w = all_weights.sum()
        if total_w > 0:
            avg_scored = float(np.average(all_scored, weights=all_weights))
            avg_conceded = float(np.average(all_conceded, weights=all_weights))
        else:
            avg_scored = 1.0
            avg_conceded = 1.0

        team_stats[team] = {
            "avg_scored": round(avg_scored, 4),
            "avg_conceded": round(avg_conceded, 4),
        }

    # --- Final Elo ratings (for inference in predictor.py) ---
    # Replay Elo over the full combined corpus (World Cup + international,
    # up to the most recent match available).
    df_with_host = add_host_flags(df)
    elo_ratings = get_final_elo_ratings(df_with_host)
    elo_ratings = {team: round(elo, 1) for team, elo in elo_ratings.items()}

    # --- Recent-form snapshot (for inference) ---
    team_form = _compute_current_form(df)

    return X, y_home, y_away, sample_weights, team_stats, elo_ratings, team_form, feature_cols


def train_models():
    """Train and export Random Forest models."""
    print("🤖 IA Pipeline Iniciado (v4 — corpus combinado + forma reciente)...")
    print("Preparando datos históricos (Mundiales + partidos internacionales)...")
    X, y_home, y_away, sample_weights, team_stats, elo_ratings, team_form, feature_cols = prepare_data()

    num_features = X.shape[1]
    print(f"Entrenando Random Forest con {len(X)} partidos históricos ({num_features} features)...")

    home_model = RandomForestRegressor(
        n_estimators=150, max_depth=6, random_state=42
    )
    away_model = RandomForestRegressor(
        n_estimators=150, max_depth=6, random_state=42
    )

    home_model.fit(X, y_home, sample_weight=sample_weights)
    away_model.fit(X, y_away, sample_weight=sample_weights)

    os.makedirs("models", exist_ok=True)

    print("Exportando modelos .pkl y archivos de features...")
    joblib.dump(home_model, "models/home_model.pkl")
    joblib.dump(away_model, "models/away_model.pkl")

    with open("models/team_stats.json", "w", encoding="utf-8") as f:
        json.dump(team_stats, f, indent=4)

    with open("models/elo_ratings.json", "w", encoding="utf-8") as f:
        json.dump(elo_ratings, f, indent=4)

    with open("models/team_form.json", "w", encoding="utf-8") as f:
        json.dump(team_form, f, indent=4)

    with open("models/feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=4)

    import pandas as pd
    if os.path.exists("data-worldcup/squad_values.csv"):
        sv_df = pd.read_csv("data-worldcup/squad_values.csv")
        sv_2026 = sv_df[sv_df["year"] == 2026]
        sv_dict = dict(zip(sv_2026["team_name"], sv_2026["squad_value_eur"].astype(int)))
        with open("models/squad_values_2026.json", "w", encoding="utf-8") as f:
            json.dump(sv_dict, f, indent=4)

    print(f"✅ ¡Entrenamiento completado! {len(team_stats)} equipos indexados.")
    print(f"   Modelos:     models/home_model.pkl, models/away_model.pkl")
    print(f"   Stats:       models/team_stats.json")
    print(f"   Forma:       models/team_form.json ({len(team_form)} equipos)")
    print(f"   Features:    models/feature_columns.json ({len(feature_cols)} features)")
    print(f"   Elo ratings: models/elo_ratings.json ({len(elo_ratings)} equipos)")
    print(f"   Top 5 Elo:")
    top5 = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)[:5]
    for team, elo in top5:
        print(f"     {team:25s} {elo:.1f}")


if __name__ == "__main__":
    train_models()
