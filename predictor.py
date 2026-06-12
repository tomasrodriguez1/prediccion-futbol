"""
predictor.py

This module contains the mathematical and statistical logic to predict football match outcomes.
It uses the Poisson Distribution (via scipy.stats.poisson) to model the number of goals 
scored by each team in a match.

Poisson Model Concept:
Football goals can be modeled as rare, independent events occurring at a constant rate.
For a match between Team A (Home/Designated) and Team B (Away/Designated):
- Expected Goals for Team A (lambda_A) = Attack_A * Defense_B * Tournament_Average_Goals
- Expected Goals for Team B (lambda_B) = Attack_B * Defense_A * Tournament_Average_Goals

The probability of Team A scoring x goals and Team B scoring y goals is:
P(X=x, Y=y) = Poisson(x; lambda_A) * Poisson(y; lambda_B)
"""

import numpy as np
from scipy.stats import poisson
import joblib
import json
import os
from dixon_coles import dc_score_matrix
from data_prep import (
    CONFEDERATION_MAP, CONFEDERATION_WIN_RATES,
    NEUTRAL_PRIOR_SCORED, NEUTRAL_PRIOR_POINTS,
)
from team_names import normalize_team_name

WC2026_HOSTS = {"USA", "Mexico", "Canada"}

# Seed ratings represent prior knowledge of team strengths. 
# This prevents new teams or teams with no games played from defaulting to identical 1.0 ratings.
# Higher attack = scores more goals; Lower defense = concedes fewer goals.
SEED_RATINGS = {
    "Argentina": {"attack": 1.45, "defense": 0.65},
    "France": {"attack": 1.40, "defense": 0.70},
    "Brazil": {"attack": 1.35, "defense": 0.75},
    "England": {"attack": 1.30, "defense": 0.75},
    "Spain": {"attack": 1.38, "defense": 0.68},
    "Portugal": {"attack": 1.25, "defense": 0.80},
    "Netherlands": {"attack": 1.20, "defense": 0.85},
    "Germany": {"attack": 1.25, "defense": 0.90},
    "Italy": {"attack": 1.10, "defense": 0.75},
    "Uruguay": {"attack": 1.20, "defense": 0.80},
    "Croatia": {"attack": 1.15, "defense": 0.85},
    "Morocco": {"attack": 1.05, "defense": 0.80},
    "Colombia": {"attack": 1.18, "defense": 0.82},
    "USA": {"attack": 1.10, "defense": 0.95},
    "Mexico": {"attack": 1.05, "defense": 1.00},
    "Canada": {"attack": 1.05, "defense": 1.05},
    "Japan": {"attack": 1.15, "defense": 0.90},
    "South Korea": {"attack": 1.10, "defense": 0.95},
    "Senegal": {"attack": 1.05, "defense": 0.90},
    "Ecuador": {"attack": 1.05, "defense": 0.85},
    "Belgium": {"attack": 1.18, "defense": 0.92},
    "Switzerland": {"attack": 1.08, "defense": 0.88},
    "Denmark": {"attack": 1.05, "defense": 0.90},
    "Ukraine": {"attack": 1.05, "defense": 0.95},
    "Austria": {"attack": 1.10, "defense": 0.90},
    "Turkey": {"attack": 1.12, "defense": 0.98},
    "Saudi Arabia": {"attack": 0.85, "defense": 1.15},
    "Australia": {"attack": 0.95, "defense": 1.05},
    "Iran": {"attack": 0.95, "defense": 1.00},
    "South Africa": {"attack": 0.90, "defense": 1.10},
    "Cameroon": {"attack": 0.95, "defense": 1.08},
}

DEFAULT_RATING = {"attack": 1.0, "defense": 1.0}
DEFAULT_TOURNAMENT_AVG = 1.35  # Average goals scored per team per match (~2.7 total goals per game)

def calculate_team_ratings(finished_matches, prior_weight=3.0):
    """
    Computes attack and defense ratings for all teams based on historical match data.
    Implements a Bayesian-like shrink toward seeded ratings to smooth out teams with 
    very few games played.
    
    :param finished_matches: List of match dictionaries containing results.
    :param prior_weight: The weight given to the seed ratings (equivalent to prior games).
    :return: Dict of team ratings {team_name: {"attack": float, "defense": float}}
    """
    # 1. If no finished matches, return the seed ratings directly
    if not finished_matches:
        ratings = {}
        for team, seed in SEED_RATINGS.items():
            ratings[team] = seed.copy()
        return ratings
    
    # 2. Accumulate goals scored, goals conceded, and games played per team
    stats = {}
    total_goals = 0
    total_games_doubled = 0
    
    for match in finished_matches:
        home_team = normalize_team_name(match.get("homeTeam", {}).get("name"))
        away_team = normalize_team_name(match.get("awayTeam", {}).get("name"))
        score = match.get("score", {})
        
        # We need the actual full-time score
        full_time = score.get("fullTime", {})
        home_goals = full_time.get("home")
        away_goals = full_time.get("away")
        
        if home_goals is None or away_goals is None:
            continue
            
        # Initialize stats if not present
        for team in [home_team, away_team]:
            if team not in stats:
                stats[team] = {"goals_scored": 0, "goals_conceded": 0, "games_played": 0}
                
        stats[home_team]["goals_scored"] += home_goals
        stats[home_team]["goals_conceded"] += away_goals
        stats[home_team]["games_played"] += 1
        
        stats[away_team]["goals_scored"] += away_goals
        stats[away_team]["goals_conceded"] += home_goals
        stats[away_team]["games_played"] += 1
        
        total_goals += (home_goals + away_goals)
        total_games_doubled += 2

    # 3. Calculate average goals per team per match in this dataset
    if total_games_doubled > 0:
        dataset_avg_goals = total_goals / total_games_doubled
    else:
        dataset_avg_goals = DEFAULT_TOURNAMENT_AVG

    # 4. Calculate ratings combining empirical data and prior seeds (Bayesian Shrinkage)
    ratings = {}
    
    # Gather all teams from stats and seed ratings to ensure complete coverage
    all_teams = set(stats.keys()).union(SEED_RATINGS.keys())
    
    for team in all_teams:
        team_stats = stats.get(team, {"goals_scored": 0, "goals_conceded": 0, "games_played": 0})
        seed = SEED_RATINGS.get(team, DEFAULT_RATING)
        
        games = team_stats["games_played"]
        goals_s = team_stats["goals_scored"]
        goals_c = team_stats["goals_conceded"]
        
        # Bayesian Shrinkage Formula:
        # Attack = (Empirical Goals Scored + Weight * Seed Attack) / (Games Played * Avg Goals + Weight)
        # Defense = (Empirical Goals Conceded + Weight * Seed Defense) / (Games Played * Avg Goals + Weight)
        denom = (games * dataset_avg_goals) + prior_weight
        
        attack = (goals_s + (prior_weight * seed["attack"])) / denom
        defense = (goals_c + (prior_weight * seed["defense"])) / denom
        
        ratings[team] = {
            "attack": round(attack, 3),
            "defense": round(defense, 3),
            "games_played": games
        }
        
    return ratings


def _conf_wr(team: str) -> float:
    return CONFEDERATION_WIN_RATES.get(CONFEDERATION_MAP.get(team, "UEFA"), 0.294)


def load_ml_config() -> dict:
    """Load RF models and lookup tables for inference."""
    cfg: dict = {"ml_ready": False}
    required = ["models/home_model.pkl", "models/team_stats.json", "models/feature_columns.json"]
    if not all(os.path.exists(p) for p in required):
        return cfg

    cfg["home_model"] = joblib.load("models/home_model.pkl")
    cfg["away_model"] = joblib.load("models/away_model.pkl")
    with open("models/team_stats.json", encoding="utf-8") as f:
        cfg["team_stats"] = json.load(f)
    with open("models/feature_columns.json", encoding="utf-8") as f:
        cfg["feature_columns"] = json.load(f)
    cfg["elo_ratings"] = {}
    if os.path.exists("models/elo_ratings.json"):
        with open("models/elo_ratings.json", encoding="utf-8") as f:
            cfg["elo_ratings"] = json.load(f)
    cfg["team_form"] = {}
    if os.path.exists("models/team_form.json"):
        with open("models/team_form.json", encoding="utf-8") as f:
            cfg["team_form"] = json.load(f)
    cfg["sv_2026"] = {}
    if os.path.exists("models/squad_values_2026.json"):
        with open("models/squad_values_2026.json", encoding="utf-8") as f:
            cfg["sv_2026"] = json.load(f)
    cfg["ml_ready"] = True
    return cfg


def build_ml_feature_values(home_team: str, away_team: str, cfg: dict) -> dict:
    """Build the feature dict for one match (keys match models/feature_columns.json)."""
    home_team = normalize_team_name(home_team)
    away_team = normalize_team_name(away_team)

    ts = cfg["team_stats"]
    elo = cfg["elo_ratings"]
    team_form = cfg.get("team_form", {})
    sv = cfg.get("sv_2026", {})

    h_stat = ts.get(home_team, {"avg_scored": 1.0, "avg_conceded": 1.0})
    a_stat = ts.get(away_team, {"avg_scored": 1.0, "avg_conceded": 1.0})

    neutral_form = {
        "form_scored": NEUTRAL_PRIOR_SCORED,
        "form_conceded": NEUTRAL_PRIOR_SCORED,
        "form_points": NEUTRAL_PRIOR_POINTS,
    }
    h_form = team_form.get(home_team, neutral_form)
    a_form = team_form.get(away_team, neutral_form)

    elo_home = elo.get(home_team, 1500.0)
    elo_away = elo.get(away_team, 1500.0)
    elo_diff = elo_home - elo_away

    sv_home = sv_away = sv_ratio = None
    if sv:
        sv_median = float(np.median(list(sv.values()))) if sv else 1_000_000
        sv_home_eur = sv.get(home_team, sv_median)
        sv_away_eur = sv.get(away_team, sv_median)
        sv_home = sv_home_eur / 1e6
        sv_away = sv_away_eur / 1e6
        sv_ratio = sv_home / max(sv_away, 1.0)

    return {
        "h_avg_scored": h_stat["avg_scored"], "h_avg_conceded": h_stat["avg_conceded"],
        "a_avg_scored": a_stat["avg_scored"], "a_avg_conceded": a_stat["avg_conceded"],
        "h_form_scored": h_form["form_scored"], "h_form_conceded": h_form["form_conceded"],
        "h_form_points": h_form["form_points"],
        "a_form_scored": a_form["form_scored"], "a_form_conceded": a_form["form_conceded"],
        "a_form_points": a_form["form_points"],
        "elo_home": elo_home, "elo_away": elo_away, "elo_diff": elo_diff,
        "host_home": 1 if home_team in WC2026_HOSTS else 0,
        "host_away": 1 if away_team in WC2026_HOSTS else 0,
        "is_world_cup": 1,
        "sv_home": sv_home, "sv_away": sv_away, "sv_ratio": sv_ratio,
        "conf_strength_home": _conf_wr(home_team), "conf_strength_away": _conf_wr(away_team),
    }


def build_ml_feature_matrix(home_team: str, away_team: str, cfg: dict) -> np.ndarray:
    """Return a (1, n_features) matrix ordered by models/feature_columns.json."""
    values = build_ml_feature_values(home_team, away_team, cfg)
    row = [values[col] for col in cfg["feature_columns"]]
    return np.array([row])


def predict_match(home_team, away_team, team_ratings, tournament_avg=DEFAULT_TOURNAMENT_AVG, max_goals=8):
    """
    Predicts the outcome probability of a match using the Poisson distribution.
    Now enhanced with Random Forest Machine Learning for Expected Goals (xG) calculation.
    """
    home_team = normalize_team_name(home_team)
    away_team = normalize_team_name(away_team)

    ml_used = False
    lambda_home, lambda_away = 1.0, 1.0
    # Dixon-Coles disabled in production (rho=0 == independent Poisson).
    # Walk-forward backtesting showed the correction slightly worsened
    # log-loss and accuracy on World Cup data, where goal dependence is ~0.
    # dixon_coles.py is kept for reuse on league data, where it does help.
    rho = 0.0

    try:
        cfg = load_ml_config()
        if cfg["ml_ready"]:
            features = build_ml_feature_matrix(home_team, away_team, cfg)
            home_model = cfg["home_model"]
            away_model = cfg["away_model"]

            n_expected = home_model.n_features_in_
            if features.shape[1] != n_expected:
                raise ValueError(
                    f"Feature count mismatch: built {features.shape[1]}, "
                    f"model expects {n_expected}. Run train_model.py to retrain."
                )

            lambda_home = float(home_model.predict(features)[0])
            lambda_away = float(away_model.predict(features)[0])

            # Ensure lambdas are reasonable strictly positive
            lambda_home = max(0.1, lambda_home)
            lambda_away = max(0.1, lambda_away)
            ml_used = True
    except Exception as e:
        print(f"Failed to load ML models: {e}")
        
    if not ml_used:
        # Get ratings (fallback to seed ratings or default if missing)
        home_rating = team_ratings.get(home_team, SEED_RATINGS.get(home_team, DEFAULT_RATING))
        away_rating = team_ratings.get(away_team, SEED_RATINGS.get(away_team, DEFAULT_RATING))
        
        home_attack = home_rating.get("attack", 1.0)
        home_defense = home_rating.get("defense", 1.0)
        
        away_attack = away_rating.get("attack", 1.0)
        away_defense = away_rating.get("defense", 1.0)
        
        # Calculate Poisson lambdas (expected goals) using traditional stats
        lambda_home = home_attack * away_defense * tournament_avg
        lambda_away = away_attack * home_defense * tournament_avg
    
    # rho is fixed to 0, so this is an independent Poisson score matrix.
    score_matrix = dc_score_matrix(lambda_home, lambda_away, rho, max_goals)
        
    # Calculate W / D / L probabilities
    home_win_prob = 0.0
    away_win_prob = 0.0
    draw_prob = 0.0
    
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            prob = score_matrix[h, a]
            if h > a:
                home_win_prob += prob
            elif h < a:
                away_win_prob += prob
            else:
                draw_prob += prob
                
    # Determine the top 5 most likely exact scores
    scores_list = []
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            scores_list.append(((h, a), float(score_matrix[h, a])))
            
    scores_list.sort(key=lambda x: x[1], reverse=True)
    top_scores = scores_list[:5]
    
    return {
        "home_win_prob": round(home_win_prob, 4),
        "away_win_prob": round(away_win_prob, 4),
        "draw_prob": round(draw_prob, 4),
        "expected_home_goals": round(lambda_home, 3),
        "expected_away_goals": round(lambda_away, 3),
        "score_matrix": score_matrix,
        "top_scores": top_scores,
        "ml_used": ml_used
    }
