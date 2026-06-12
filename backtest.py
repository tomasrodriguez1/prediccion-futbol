"""
backtest.py

Walk-forward temporal backtest for the World Cup prediction system.

Methodology
-----------
1. World Cup matches are grouped by tournament (WC-1930, WC-1934, ..., WC-2022).
2. For each tournament T(i), models are trained on data STRICTLY BEFORE T(i)
   started, and evaluated on T(i)'s matches. The very first tournament (1930)
   is skipped (no training data).
3. Predictions are probabilistic: P(home win), P(draw), P(away win), derived
   from the Poisson score-matrix (rho=0, matching production — see
   predictor.predict_match).
4. Metrics are aggregated over all out-of-sample predictions.

Metrics
-------
- **RPS** (Ranked Probability Score): proper scoring rule for ordinal outcomes.
- **Log-loss** (multiclass): standard probabilistic calibration metric.
- **Brier score**: mean squared error of probability vs one-hot outcome.
- **Accuracy**: fraction of correct result predictions (secondary).

Three models compared:
- **BASELINE**: always predicts the historical frequency of home-win/draw/away-win
  from the World Cup matches seen so far.
- **ACTUAL**: replicates the CURRENT production approach — trained only on
  World Cup matches (1930-2022), 14 features (goal averages, Elo, host flags,
  squad values, confederation strength). No international data, no form.
- **NUEVO**: trained on the COMBINED corpus (World Cup + international
  matches, restricted to >= 1990), 21 features = ACTUAL's 14 + recent-form
  (last 5 matches: goals scored/conceded + points) + an `is_world_cup` flag.

NUEVO requires enough post-1990 international history to train on, so it is
only evaluated for tournaments where the pre-cutoff training set has at least
``MIN_FOLD_SIZE`` rows (in practice WC-1998 onward). BASELINE and ACTUAL are
evaluated for every tournament (1934 onward) for reference, plus a head-to-head
table over the common folds where all three are available.

How to run
----------
    .venv/bin/python backtest.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import poisson

from data_prep import (
    load_combined_corpus, build_asof_features, add_squad_values,
    add_confederation_features, compute_temporal_weight,
)
from train_model import BASE_FEATURE_COLS, MIN_TRAIN_DATE


# Feature list replicating the CURRENT production model (14 features,
# World-Cup-only training data).
FEATURES_ACTUAL = [
    "h_avg_scored", "h_avg_conceded", "a_avg_scored", "a_avg_conceded",
    "elo_home", "elo_away", "elo_diff",
    "host_home", "host_away",
    "sv_home", "sv_away", "sv_ratio",
    "conf_strength_home", "conf_strength_away",
]

# Feature list for the NEW model: ACTUAL's base + form features + is_world_cup,
# trained on the combined (WC + international) corpus.
FEATURES_NEW = BASE_FEATURE_COLS + ["sv_home", "sv_away", "sv_ratio", "conf_strength_home", "conf_strength_away"]

# Minimum number of post-1990 training rows required to fit the NEW model
# for a given fold.
MIN_FOLD_SIZE = 200

HALF_LIFE = 8.0


# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------

def _rps(p: np.ndarray, y: np.ndarray) -> float:
    """Ranked Probability Score for a single observation."""
    cum_p = np.cumsum(p)
    cum_y = np.cumsum(y)
    return float(np.mean((cum_p - cum_y) ** 2))


def _logloss(p: np.ndarray, y: np.ndarray, eps: float = 1e-15) -> float:
    """Log-loss for a single observation."""
    p = np.clip(p, eps, 1 - eps)
    return float(-np.sum(y * np.log(p)))


def _brier(p: np.ndarray, y: np.ndarray) -> float:
    """Brier score for a single observation."""
    return float(np.mean((p - y) ** 2))


# ---------------------------------------------------------------------------
# POISSON OUTCOME PROBABILITIES
# ---------------------------------------------------------------------------

def poisson_outcome_probs(lambda_home: float, lambda_away: float, max_goals: int = 8):
    """Return (P_home_win, P_draw, P_away_win) from a Poisson score matrix."""
    lambda_home = max(0.1, lambda_home)
    lambda_away = max(0.1, lambda_away)

    h_probs = poisson.pmf(np.arange(max_goals + 1), lambda_home)
    a_probs = poisson.pmf(np.arange(max_goals + 1), lambda_away)
    matrix = np.outer(h_probs, a_probs)
    matrix /= matrix.sum()

    home_win = float(np.sum(np.tril(matrix, -1)))
    draw = float(np.sum(np.diag(matrix)))
    away_win = float(np.sum(np.triu(matrix, 1)))
    return np.array([home_win, draw, away_win])


def _one_hot_outcome(home_goals: int, away_goals: int) -> np.ndarray:
    if home_goals > away_goals:
        return np.array([1, 0, 0])
    elif home_goals == away_goals:
        return np.array([0, 1, 0])
    return np.array([0, 0, 1])


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _recompute_weight(df: pd.DataFrame, ref_date, half_life: float = HALF_LIFE) -> pd.DataFrame:
    """Return a copy of *df* with `weight` recomputed relative to *ref_date*.

    Matches AFTER ref_date (shouldn't normally happen for training data) get
    weight >= 1; this mirrors the exponential-decay formula used everywhere
    else, just re-anchored to the fold's cutoff instead of "today".
    """
    df = df.copy()
    age_years = (ref_date - df["match_date"]).dt.days / 365.25
    df["weight"] = np.exp(-np.log(2) * age_years / half_life)
    return df


def _fit_predict(train_df, test_df, feature_cols):
    """Fit home/away RandomForest regressors and predict expected goals for test_df."""
    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values
    w_train = train_df["weight"].values

    home_model = RandomForestRegressor(n_estimators=150, max_depth=6, random_state=42)
    away_model = RandomForestRegressor(n_estimators=150, max_depth=6, random_state=42)
    home_model.fit(X_train, train_df["home_team_score"].values, sample_weight=w_train)
    away_model.fit(X_train, train_df["away_team_score"].values, sample_weight=w_train)

    pred_home = home_model.predict(X_test)
    pred_away = away_model.predict(X_test)
    return pred_home, pred_away


# ---------------------------------------------------------------------------
# WALK-FORWARD BACKTEST
# ---------------------------------------------------------------------------

def backtest():
    print("=" * 80)
    print("⚽ WALK-FORWARD BACKTEST — World Cup Prediction System")
    print("   Comparing: BASELINE vs ACTUAL (WC-only) vs NUEVO (WC + internacional + forma)")
    print("=" * 80)

    # -- 1. Load combined corpus -------------------------------------------
    combined = load_combined_corpus()
    wc = combined[combined["is_world_cup"] == 1].copy()
    wc["tournament_year"] = wc["tournament_id"].str.extract(r"(\d{4})").astype(int)
    tournaments = sorted(wc["tournament_year"].unique())

    print(f"\nWorld Cup tournaments found: {tournaments}")
    print(f"WC matches: {len(wc)} | Combined corpus: {len(combined)} matches "
          f"({combined['match_date'].min().date()} -> {combined['match_date'].max().date()})")
    print(f"Features ACTUAL ({len(FEATURES_ACTUAL)}): {FEATURES_ACTUAL}")
    print(f"Features NUEVO  ({len(FEATURES_NEW)}): {FEATURES_NEW}")

    all_preds = []  # one entry per (tournament, match): p_baseline, p_actual, p_new (optional), y_true

    for i, test_year in enumerate(tournaments):
        if i == 0:
            continue  # WC-1930: nothing to train on

        test_matches = wc[wc["tournament_year"] == test_year]
        start_date = test_matches["match_date"].min()
        end_date = test_matches["match_date"].max()

        # -- BASELINE: historical frequency from WC matches before this fold --
        wc_train_raw = wc[wc["match_date"] < start_date]
        n_train_wc = len(wc_train_raw)
        hw = (wc_train_raw["home_team_score"] > wc_train_raw["away_team_score"]).sum()
        dr = (wc_train_raw["home_team_score"] == wc_train_raw["away_team_score"]).sum()
        aw = n_train_wc - hw - dr
        baseline = np.array([hw / n_train_wc, dr / n_train_wc, aw / n_train_wc])

        # -- ACTUAL: WC-only training data, 14 features ------------------------
        wc_subset = wc[wc["match_date"] <= end_date]
        wc_subset = _recompute_weight(wc_subset, ref_date=start_date)
        wc_featured = build_asof_features(wc_subset)
        wc_featured = add_squad_values(wc_featured)
        wc_featured = add_confederation_features(wc_featured)

        train_actual = wc_featured[wc_featured["match_date"] < start_date]
        test_actual = wc_featured[wc_featured["match_date"] >= start_date]

        pred_home_actual, pred_away_actual = _fit_predict(train_actual, test_actual, FEATURES_ACTUAL)

        # -- NUEVO: combined corpus (>= MIN_TRAIN_DATE), 21 features -----------
        new_subset = combined[combined["match_date"] <= end_date]
        new_subset = _recompute_weight(new_subset, ref_date=start_date)
        new_featured = build_asof_features(new_subset)
        new_featured = add_squad_values(new_featured)
        new_featured = add_confederation_features(new_featured)

        train_new = new_featured[
            (new_featured["match_date"] < start_date) & (new_featured["match_date"] >= MIN_TRAIN_DATE)
        ]
        test_new = new_featured[
            (new_featured["is_world_cup"] == 1) & (new_featured["match_date"] >= start_date)
        ]

        has_new = len(train_new) >= MIN_FOLD_SIZE
        if has_new:
            pred_home_new, pred_away_new = _fit_predict(train_new, test_new, FEATURES_NEW)

        # -- Evaluate each test match -------------------------------------
        n_test = len(test_actual)
        for j in range(n_test):
            row = test_actual.iloc[j]
            y_true = _one_hot_outcome(int(row["home_team_score"]), int(row["away_team_score"]))

            p_actual = poisson_outcome_probs(pred_home_actual[j], pred_away_actual[j])

            entry = {
                "tournament": test_year,
                "home": row["home_team_name"],
                "away": row["away_team_name"],
                "actual": f"{int(row['home_team_score'])}-{int(row['away_team_score'])}",
                "p_baseline": baseline,
                "p_actual": p_actual,
                "y_true": y_true,
            }
            if has_new:
                entry["p_new"] = poisson_outcome_probs(pred_home_new[j], pred_away_new[j])

            all_preds.append(entry)

        new_status = f"NUEVO ({len(train_new):>5d} filas >= {MIN_TRAIN_DATE})" if has_new \
            else f"NUEVO omitido ({len(train_new)} filas < {MIN_FOLD_SIZE})"
        print(f"  WC {test_year}: ACTUAL train={len(train_actual):>4d} WC | {new_status} "
              f"| test={n_test:>3d} matches")

    # -- 2. Aggregate metrics: ACTUAL vs BASELINE (all folds) --------------
    _print_table(
        "ALL FOLDS (WC-1934 .. WC-2022)",
        all_preds,
        [("BASELINE", "p_baseline"), ("ACTUAL", "p_actual")],
    )

    # -- 3. Aggregate metrics: head-to-head on common folds (NUEVO available)
    common_preds = [p for p in all_preds if "p_new" in p]
    if common_preds:
        common_years = sorted(set(p["tournament"] for p in common_preds))
        _print_table(
            f"COMMON FOLDS WITH NUEVO ({common_years})",
            common_preds,
            [("BASELINE", "p_baseline"), ("ACTUAL", "p_actual"), ("NUEVO", "p_new")],
        )

        # Per-tournament breakdown (ACTUAL vs NUEVO) to see where the diff comes from.
        print(f"\n{'-' * 95}")
        print(f"{'Per-tournament':<12s}{'n':>5s}{'RPS_act':>10s}{'RPS_new':>10s}"
              f"{'Acc_act':>10s}{'Acc_new':>10s}")
        print(f"{'-' * 95}")
        for year in common_years:
            yr_preds = [p for p in common_preds if p["tournament"] == year]
            y_true = [p["y_true"] for p in yr_preds]
            rps_act = np.mean([_rps(p["p_actual"], y) for p, y in zip(yr_preds, y_true)])
            rps_new = np.mean([_rps(p["p_new"], y) for p, y in zip(yr_preds, y_true)])
            acc_act = np.mean([np.argmax(p["p_actual"]) == np.argmax(y) for p, y in zip(yr_preds, y_true)])
            acc_new = np.mean([np.argmax(p["p_new"]) == np.argmax(y) for p, y in zip(yr_preds, y_true)])
            print(f"{year:<12d}{len(yr_preds):>5d}{rps_act:>10.4f}{rps_new:>10.4f}"
                  f"{acc_act:>10.4f}{acc_new:>10.4f}")

        # Same comparison excluding the smallest (least stable) fold (WC-1990, 215 rows).
        stable_preds = [p for p in common_preds if p["tournament"] != 1990]
        _print_table(
            f"COMMON FOLDS WITH NUEVO, EXCLUDING WC-1990 (small training set)",
            stable_preds,
            [("BASELINE", "p_baseline"), ("ACTUAL", "p_actual"), ("NUEVO", "p_new")],
        )
    else:
        print("\n[AVISO] Ningún fold tuvo suficientes datos post-1990 para entrenar NUEVO.")

    print(f"\nTotal out-of-sample predictions: {len(all_preds)}")
    print(f"Tournaments tested: {sorted(set(p['tournament'] for p in all_preds))}")

    return all_preds


def _print_table(title, preds, models):
    """Print RPS / log-loss / Brier / accuracy for each (label, key) in *models*."""
    print(f"\n{'=' * 95}")
    print(title)
    print(f"{'=' * 95}")
    header = f"{'METRIC':<25s}" + "".join(f"{label:>16s}" for label, _ in models)
    print(header)
    print(f"{'-' * 95}")

    y_true = [p["y_true"] for p in preds]

    rows = [
        ("RPS (lower better)", _rps),
        ("Log-loss (lower better)", _logloss),
        ("Brier (lower better)", _brier),
    ]
    for name, fn in rows:
        line = f"{name:<25s}"
        for _, key in models:
            val = np.mean([fn(p[key], y) for p, y in zip(preds, y_true)])
            line += f"{val:>16.4f}"
        print(line)

    line = f"{'Accuracy (higher better)':<25s}"
    for _, key in models:
        acc = np.mean([np.argmax(p[key]) == np.argmax(y) for p, y in zip(preds, y_true)])
        line += f"{acc:>16.4f}"
    print(line)
    print(f"{'=' * 95}")
    print(f"Predictions: {len(preds)}")


# ---------------------------------------------------------------------------
# WEIGHT-SCHEME COMPARISON (recency boost vs half-life)
# ---------------------------------------------------------------------------

# Candidate weighting schemes: (label, half_life, recent_boost, window_days)
WEIGHT_SCHEMES = [
    ("HL8 (actual)",       8.0, 1.0,   365),
    ("HL8 +boost x2/12m",  8.0, 2.0,   365),
    ("HL8 +boost x3/12m",  8.0, 3.0,   365),
    ("HL8 +boost x5/6m",   8.0, 5.0,   180),
    ("HL4 sin boost",      4.0, 1.0,   365),
    ("HL2 sin boost",      2.0, 1.0,   365),
]


def backtest_weights():
    """Walk-forward comparison of temporal-weight schemes for the NUEVO model.

    For each World Cup fold (1998..2022) and each weight scheme, weights are
    re-anchored to the fold cutoff (so "recent" means recent relative to that
    tournament), features are rebuilt, the RF is fit, and out-of-sample
    probabilistic metrics are aggregated. Helps pick the scheme that makes
    recent results count more without hurting calibration.
    """
    print("=" * 95)
    print("⚖️  COMPARACIÓN DE ESQUEMAS DE PESO TEMPORAL (modelo NUEVO, 21 features)")
    print("=" * 95)

    combined = load_combined_corpus()
    wc = combined[combined["is_world_cup"] == 1].copy()
    wc["tournament_year"] = wc["tournament_id"].str.extract(r"(\d{4})").astype(int)
    tournaments = sorted(wc["tournament_year"].unique())

    # Per-scheme accumulator of out-of-sample predictions.
    preds_by_scheme = {label: [] for label, *_ in WEIGHT_SCHEMES}

    for test_year in tournaments:
        test_matches = wc[wc["tournament_year"] == test_year]
        start_date = test_matches["match_date"].min()
        end_date = test_matches["match_date"].max()

        new_subset = combined[combined["match_date"] <= end_date].copy()

        for label, half_life, boost, window in WEIGHT_SCHEMES:
            scheme_df = new_subset.copy()
            scheme_df["weight"] = compute_temporal_weight(
                scheme_df["match_date"], ref_date=start_date,
                half_life=half_life, recent_boost=boost, recent_window_days=window,
            )
            featured = build_asof_features(scheme_df)
            featured = add_squad_values(featured)
            featured = add_confederation_features(featured)

            train = featured[
                (featured["match_date"] < start_date) & (featured["match_date"] >= MIN_TRAIN_DATE)
            ]
            test = featured[
                (featured["is_world_cup"] == 1) & (featured["match_date"] >= start_date)
            ]
            if len(train) < MIN_FOLD_SIZE:
                continue

            pred_home, pred_away = _fit_predict(train, test, FEATURES_NEW)
            for j in range(len(test)):
                row = test.iloc[j]
                y_true = _one_hot_outcome(int(row["home_team_score"]), int(row["away_team_score"]))
                p = poisson_outcome_probs(pred_home[j], pred_away[j])
                preds_by_scheme[label].append((p, y_true))

        print(f"  WC {test_year}: evaluado bajo {len(WEIGHT_SCHEMES)} esquemas")

    # -- Aggregate metrics per scheme -------------------------------------
    print(f"\n{'=' * 95}")
    print(f"{'ESQUEMA':<22s}{'RPS':>12s}{'LogLoss':>12s}{'Brier':>12s}{'Accuracy':>12s}{'n':>8s}")
    print(f"{'-' * 95}")

    results = {}
    for label, *_ in WEIGHT_SCHEMES:
        preds = preds_by_scheme[label]
        if not preds:
            continue
        rps = np.mean([_rps(p, y) for p, y in preds])
        ll = np.mean([_logloss(p, y) for p, y in preds])
        br = np.mean([_brier(p, y) for p, y in preds])
        acc = np.mean([np.argmax(p) == np.argmax(y) for p, y in preds])
        results[label] = (rps, ll, br, acc)
        print(f"{label:<22s}{rps:>12.4f}{ll:>12.4f}{br:>12.4f}{acc:>12.4f}{len(preds):>8d}")

    print(f"{'=' * 95}")
    if results:
        best_rps = min(results, key=lambda k: results[k][0])
        best_ll = min(results, key=lambda k: results[k][1])
        print(f"Mejor RPS:      {best_rps}")
        print(f"Mejor Log-loss: {best_ll}")
    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "weights":
        backtest_weights()
    else:
        backtest()
