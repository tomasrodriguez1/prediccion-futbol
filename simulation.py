"""
simulation.py

Monte Carlo simulation of FIFA World Cup 2026.
Runs N tournament simulations to estimate each team's probability
of winning, reaching the final, semifinals, and quarterfinals.

WC 2026 format:
  - 48 teams in 12 groups of 4
  - Top 2 from each group qualify automatically (24 teams)
  - Best 8 third-placed teams also qualify (8 teams)
  - 32 teams enter knockout: R32 → R16 → QF → SF → Final

Groups are loaded from data-worldcup/wc2026_schedule.json at import time.
"""

import json
import random
import numpy as np
from collections import defaultdict

from predictor import load_ml_config, build_ml_feature_values
from team_names import normalize_team_name, is_placeholder_team

WC2026_SCHEDULE_PATH = "data-worldcup/wc2026_schedule.json"


def load_wc2026_groups(path: str = WC2026_SCHEDULE_PATH) -> dict[str, list[str]]:
    """Extract group-stage teams from the official WC 2026 schedule JSON."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    groups: dict[str, set[str]] = {}
    for match in data.get("matches", []):
        group = match.get("group", "")
        if not group or "Group" not in str(group):
            continue
        gname = str(group).replace("Group ", "").strip()
        for key in ("team1", "team2"):
            team = match.get(key)
            if team and not is_placeholder_team(team):
                groups.setdefault(gname, set()).add(normalize_team_name(team))

    return {g: sorted(teams) for g, teams in sorted(groups.items())}


def load_wc2026_bracket(path: str = WC2026_SCHEDULE_PATH) -> dict:
    """Parse the official knockout bracket from the schedule JSON.

    Slots in Round-of-32 matches are encoded as:
      - "1A" → winner of group A
      - "2B" → runner-up of group B
      - "3A/B/C/D/F" → a best-third from one of the listed groups
    Later rounds reference previous winners as "W73", "W101", etc.

    Returns
    -------
    dict with keys "Round of 32", "Round of 16", "Quarter-final",
    "Semi-final", "Final"; each a list of (match_num, slot1, slot2).
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    rounds = {"Round of 32": [], "Round of 16": [],
              "Quarter-final": [], "Semi-final": [], "Final": []}
    for match in data.get("matches", []):
        rnd = match.get("round", "")
        if rnd in rounds:
            num = match.get("num", 0)
            rounds[rnd].append((num, match["team1"], match["team2"]))

    for rnd in rounds:
        rounds[rnd].sort()
    return rounds


WC2026_GROUPS = load_wc2026_groups()
WC2026_BRACKET = load_wc2026_bracket()
ALL_TEAMS = [team for group in WC2026_GROUPS.values() for team in group]

# ---------------------------------------------------------------------------
# MODEL PRE-LOADING  (done once — avoids disk I/O inside the simulation loop)
# ---------------------------------------------------------------------------

def _precompute_lambdas(teams: list[str], cfg: dict) -> dict:
    """Pre-compute expected goals for every ordered pair using a single batch RF call.

    Builds one feature matrix (N_pairs × n_features) and calls predict once,
    making this O(1) model calls instead of O(N²).
    """
    if not cfg["ml_ready"]:
        return {(h, a): (1.2, 1.0) for h in teams for a in teams if h != a}

    pairs = [(h, a) for h in teams for a in teams if h != a]
    feature_cols = cfg["feature_columns"]

    rows = []
    for h, a in pairs:
        values = build_ml_feature_values(h, a, cfg)
        rows.append([values[col] for col in feature_cols])

    X = np.array(rows)
    lam_h_all = np.clip(cfg["home_model"].predict(X), 0.1, None)
    lam_a_all = np.clip(cfg["away_model"].predict(X), 0.1, None)

    return {pair: (float(lh), float(la))
            for pair, lh, la in zip(pairs, lam_h_all, lam_a_all)}


# ---------------------------------------------------------------------------
# SIMULATION CORE
# ---------------------------------------------------------------------------

def _simulate_match_fast(home: str, away: str, lambdas: dict) -> tuple[str, int, int]:
    """Simulate one match using pre-computed lambdas. Draws allowed (group stage)."""
    lam_h, lam_a = lambdas.get((home, away), (1.2, 1.0))
    h_goals = int(np.random.poisson(lam_h))
    a_goals = int(np.random.poisson(lam_a))
    if h_goals > a_goals:
        winner = home
    elif a_goals > h_goals:
        winner = away
    else:
        winner = "draw"
    return winner, h_goals, a_goals


def _simulate_knockout_fast(home: str, away: str, lambdas: dict) -> str:
    """Knockout match — draw resolved by coin flip (penalties)."""
    winner, _, _ = _simulate_match_fast(home, away, lambdas)
    if winner == "draw":
        winner = random.choice([home, away])
    return winner


def _simulate_group(teams: list[str], lambdas: dict) -> tuple[list, dict, dict]:
    """Simulate round-robin group. Returns (standings, points, gd)."""
    points = {t: 0 for t in teams}
    gd     = {t: 0 for t in teams}
    gf_map = {t: 0 for t in teams}

    pairs = [(teams[i], teams[j])
             for i in range(len(teams)) for j in range(i + 1, len(teams))]

    for home, away in pairs:
        winner, hg, ag = _simulate_match_fast(home, away, lambdas)
        gf_map[home] += hg
        gf_map[away] += ag
        gd[home] += hg - ag
        gd[away] += ag - hg
        if winner == home:
            points[home] += 3
        elif winner == away:
            points[away] += 3
        else:
            points[home] += 1
            points[away] += 1

    standings = sorted(
        teams,
        key=lambda t: (points[t], gd[t], gf_map[t], random.random()),
        reverse=True
    )
    return standings, points, gd


def _assign_thirds_to_slots(third_slots: list[tuple[int, frozenset]],
                            qualified_thirds: dict[str, str]) -> dict[int, str]:
    """Assign the 8 qualified third-placed teams to bracket slots.

    Each slot only admits thirds from certain groups (e.g. "3A/B/C/D/F").
    Solved as a perfect bipartite matching via backtracking, trying the
    most-constrained slot first. FIFA's slot design guarantees a solution
    for every combination of 8 qualified groups.

    Parameters
    ----------
    third_slots : list of (match_num, allowed_groups)
    qualified_thirds : {group_letter: team_name} for the 8 best thirds

    Returns
    -------
    {match_num: team_name}
    """
    available = set(qualified_thirds.keys())
    slots = sorted(third_slots,
                   key=lambda s: len(s[1] & available))

    assignment: dict[int, str] = {}

    def _backtrack(i: int, remaining: set[str]) -> bool:
        if i == len(slots):
            return True
        num, allowed = slots[i]
        for g in sorted(allowed & remaining):
            assignment[num] = qualified_thirds[g]
            if _backtrack(i + 1, remaining - {g}):
                return True
            del assignment[num]
        return False

    if not _backtrack(0, available):
        # Should not happen with the official slot design; relax constraints.
        assignment = {}
        remaining = list(qualified_thirds.values())
        for num, _ in slots:
            assignment[num] = remaining.pop()
    return assignment


def simulate_tournament(lambdas: dict, groups: dict = None,
                        bracket: dict = None) -> dict:
    """Run one complete WC 2026 simulation following the official bracket.

    Returns {team: furthest_round} where round is one of:
    'group' | 'r32' | 'r16' | 'qf' | 'sf' | 'final' | 'winner'
    """
    if groups is None:
        groups = WC2026_GROUPS
    if bracket is None:
        bracket = WC2026_BRACKET

    tournament_teams = [t for grp in groups.values() for t in grp]
    results = {team: "group" for team in tournament_teams}

    # --- GROUP STAGE ---
    group_winners: dict[str, str] = {}
    group_runners: dict[str, str] = {}
    third_place_pool = []  # (pts, gd, tiebreak, group, team)

    for grp, teams in groups.items():
        standings, pts, gd = _simulate_group(teams, lambdas)
        group_winners[grp] = standings[0]
        group_runners[grp] = standings[1]
        results[standings[0]] = "r32"
        results[standings[1]] = "r32"
        third = standings[2]
        third_place_pool.append((pts[third], gd[third], random.random(), grp, third))

    # Best 8 third-placed teams qualify
    third_place_pool.sort(reverse=True)
    qualified_thirds = {grp: team for _, _, _, grp, team in third_place_pool[:8]}
    for team in qualified_thirds.values():
        results[team] = "r32"

    # --- ROUND OF 32 (official slots) ---
    # Pre-resolve third-place slots with the bipartite matching.
    third_slots = []
    for num, s1, s2 in bracket["Round of 32"]:
        for slot in (s1, s2):
            if slot.startswith("3"):
                allowed = frozenset(slot[1:].split("/"))
                third_slots.append((num, allowed))
    third_assignment = _assign_thirds_to_slots(third_slots, qualified_thirds)

    def _resolve_slot(slot: str, match_num: int) -> str:
        if slot.startswith("1"):
            return group_winners[slot[1:]]
        if slot.startswith("2"):
            return group_runners[slot[1:]]
        return third_assignment[match_num]

    match_winners: dict[str, str] = {}  # "W73" → team

    for num, s1, s2 in bracket["Round of 32"]:
        home = _resolve_slot(s1, num)
        away = _resolve_slot(s2, num)
        w = _simulate_knockout_fast(home, away, lambdas)
        results[w] = "r16"
        match_winners[f"W{num}"] = w

    # --- ROUND OF 16 → FINAL (official W## references) ---
    for rnd_name, result_key in [("Round of 16", "qf"), ("Quarter-final", "sf"),
                                 ("Semi-final", "final"), ("Final", "winner")]:
        for num, s1, s2 in bracket[rnd_name]:
            home = match_winners[s1]
            away = match_winners[s2]
            w = _simulate_knockout_fast(home, away, lambdas)
            results[w] = result_key
            if num:
                match_winners[f"W{num}"] = w

    return results


def run_monte_carlo(team_ratings: dict, n_simulations: int = 10_000,
                    groups: dict = None) -> dict:
    """Run N simulations. Returns per-team round-reach probabilities.

    Parameters
    ----------
    team_ratings : dict
        Unused (kept for API compatibility). Models are loaded internally.
    n_simulations : int
        Number of full tournament simulations to run.
    groups : dict, optional
        Override WC2026_GROUPS for custom bracket.

    Returns
    -------
    dict: {team: {"winner": float, "final": float, "sf": float,
                  "qf": float, "r16": float, "r32": float}}
    """
    if groups is None:
        groups = WC2026_GROUPS

    tournament_teams = [t for grp in groups.values() for t in grp]
    ROUND_ORDER = ["group", "r32", "r16", "qf", "sf", "final", "winner"]

    # Load models & pre-compute all pairwise lambdas ONCE
    cfg = load_ml_config()
    lambdas = _precompute_lambdas(tournament_teams, cfg)

    counts = {team: defaultdict(int) for team in tournament_teams}

    for _ in range(n_simulations):
        result = simulate_tournament(lambdas, groups)
        for team, reached in result.items():
            idx = ROUND_ORDER.index(reached)
            for r in ROUND_ORDER[1:idx + 1]:
                counts[team][r] += 1

    probs = {}
    for team in tournament_teams:
        probs[team] = {
            r: round(counts[team][r] / n_simulations, 4)
            for r in ["r32", "r16", "qf", "sf", "final", "winner"]
        }

    return probs
