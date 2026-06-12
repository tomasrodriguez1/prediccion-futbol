"""
dixon_coles.py

Dixon-Coles (1997) correction for the independent Poisson score matrix.

The standard Poisson model assumes home and away goals are independent.
In practice, low-scoring outcomes (especially 0-0 and 1-1 draws) are more
frequent than the independent model predicts.  Dixon & Coles introduced a
correction factor τ(x, y, λ, μ, ρ) that adjusts only the four cells where
both teams score ≤ 1 goal.

The dependency parameter ρ is typically negative in football (ρ ≈ −0.05 to
−0.15), meaning low-score draws are *more* likely than independence implies.

Key functions
-------------
- ``dc_tau``           : correction factor for a single cell (x, y)
- ``dc_score_matrix``  : full corrected score matrix (non-negative, sums to 1)
- ``dc_outcome_probs`` : (P_home, P_draw, P_away) from the corrected matrix
- ``estimate_rho``     : MLE estimation of ρ from observed data
"""

import warnings

import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize_scalar


# ---------------------------------------------------------------------------
# TAU CORRECTION FACTOR
# ---------------------------------------------------------------------------

def dc_tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """Dixon-Coles τ correction factor for the score cell (x, y).

    Parameters
    ----------
    x : int   — home goals
    y : int   — away goals
    lam : float — expected home goals (λ)
    mu  : float — expected away goals (μ)
    rho : float — dependency parameter

    Returns
    -------
    float
        Multiplicative correction.  For cells outside {0,1}×{0,1} this is 1.
    """
    if x == 0 and y == 0:
        return 1.0 - lam * mu * rho
    elif x == 0 and y == 1:
        return 1.0 + lam * rho
    elif x == 1 and y == 0:
        return 1.0 + mu * rho
    elif x == 1 and y == 1:
        return 1.0 - rho
    else:
        return 1.0


# ---------------------------------------------------------------------------
# CORRECTED SCORE MATRIX
# ---------------------------------------------------------------------------

def dc_score_matrix(
    lam: float,
    mu: float,
    rho: float,
    max_goals: int = 8,
) -> np.ndarray:
    """Build a Dixon-Coles corrected score-probability matrix.

    Steps:
      1. Start from independent Poisson outer product.
      2. Multiply each of the four low-score cells by its τ factor.
      3. Clip to ≥ 0 (safety for extreme ρ values).
      4. Renormalise so the matrix sums to 1.

    Parameters
    ----------
    lam : float      — expected home goals (λ), must be > 0
    mu  : float      — expected away goals (μ), must be > 0
    rho : float      — Dixon-Coles dependency parameter
    max_goals : int  — maximum goals per side in the grid (default 8)

    Returns
    -------
    np.ndarray of shape (max_goals+1, max_goals+1)
        Probability of each score (home_goals, away_goals).
    """
    lam = max(0.1, lam)
    mu = max(0.1, mu)

    # Independent Poisson matrix
    h_probs = poisson.pmf(np.arange(max_goals + 1), lam)
    a_probs = poisson.pmf(np.arange(max_goals + 1), mu)
    matrix = np.outer(h_probs, a_probs)

    # Apply τ correction to the four low-score cells
    for x in range(min(2, max_goals + 1)):
        for y in range(min(2, max_goals + 1)):
            matrix[x, y] *= dc_tau(x, y, lam, mu, rho)

    # Safety: clip to non-negative (extreme ρ could make a cell negative)
    matrix = np.clip(matrix, 0.0, None)

    # Renormalise
    total = matrix.sum()
    if total > 0:
        matrix /= total

    # Postcondition checks
    assert np.all(matrix >= 0), "Score matrix contains negative probabilities!"
    assert abs(matrix.sum() - 1.0) < 1e-10, f"Score matrix does not sum to 1 (sum={matrix.sum()})!"

    return matrix


# ---------------------------------------------------------------------------
# OUTCOME PROBABILITIES
# ---------------------------------------------------------------------------

def dc_outcome_probs(
    lam: float,
    mu: float,
    rho: float,
    max_goals: int = 8,
) -> np.ndarray:
    """Return (P_home_win, P_draw, P_away_win) from the DC score matrix.

    Parameters
    ----------
    lam : float — expected home goals
    mu  : float — expected away goals
    rho : float — Dixon-Coles dependency parameter

    Returns
    -------
    np.ndarray of shape (3,)
        [P_home_win, P_draw, P_away_win]
    """
    matrix = dc_score_matrix(lam, mu, rho, max_goals)
    n = max_goals + 1

    home_win = 0.0
    draw = 0.0
    away_win = 0.0

    for h in range(n):
        for a in range(n):
            prob = matrix[h, a]
            if h > a:
                home_win += prob
            elif h == a:
                draw += prob
            else:
                away_win += prob

    return np.array([home_win, draw, away_win])


# ---------------------------------------------------------------------------
# ESTIMATION OF ρ (MLE)
# ---------------------------------------------------------------------------

def estimate_rho(
    goals_home: np.ndarray,
    goals_away: np.ndarray,
    lambdas_home: np.ndarray,
    lambdas_away: np.ndarray,
    bounds: tuple[float, float] = (-0.2, 0.2),
    fallback: float = -0.1,
) -> float:
    """Estimate the Dixon-Coles ρ parameter via maximum likelihood.

    The log-likelihood of the observed scores under the DC model is:
        Σ_i  log[ Poisson(x_i; λ_i) · Poisson(y_i; μ_i) · τ(x_i, y_i, λ_i, μ_i, ρ) ]

    Since the Poisson terms don't depend on ρ, maximising ρ reduces to:
        Σ_i  log[ τ(x_i, y_i, λ_i, μ_i, ρ) ]
    subject to τ > 0 for all observations.

    Parameters
    ----------
    goals_home : array-like — observed home goals per match
    goals_away : array-like — observed away goals per match
    lambdas_home : array-like — predicted λ (expected home goals) per match
    lambdas_away : array-like — predicted μ (expected away goals) per match
    bounds : tuple — search interval for ρ
    fallback : float — returned if optimisation fails

    Returns
    -------
    float — estimated ρ
    """
    goals_home = np.asarray(goals_home, dtype=int)
    goals_away = np.asarray(goals_away, dtype=int)
    lambdas_home = np.asarray(lambdas_home, dtype=float)
    lambdas_away = np.asarray(lambdas_away, dtype=float)

    # Only matches with both goals ≤ 1 contribute to the τ term
    # (for all other matches τ = 1 and log(τ) = 0).
    mask = (goals_home <= 1) & (goals_away <= 1)
    if mask.sum() == 0:
        warnings.warn("No low-scoring matches in training data; using fallback ρ.")
        return fallback

    gh = goals_home[mask]
    ga = goals_away[mask]
    lh = lambdas_home[mask]
    la = lambdas_away[mask]

    def neg_log_lik(rho: float) -> float:
        """Negative log-likelihood of τ terms (to minimise)."""
        ll = 0.0
        for i in range(len(gh)):
            tau = dc_tau(int(gh[i]), int(ga[i]), float(lh[i]), float(la[i]), rho)
            if tau <= 0:
                return 1e12  # penalty: infeasible ρ
            ll += np.log(tau)
        return -ll

    try:
        result = minimize_scalar(
            neg_log_lik,
            bounds=bounds,
            method="bounded",
            options={"xatol": 1e-6, "maxiter": 200},
        )
        if result.success or result.fun < 1e11:
            rho_hat = float(result.x)
            # Sanity: clamp to bounds
            rho_hat = max(bounds[0], min(bounds[1], rho_hat))
            return rho_hat
        else:
            warnings.warn(
                f"ρ optimisation did not converge (status={result.message}); "
                f"using fallback ρ={fallback}."
            )
            return fallback
    except Exception as e:
        warnings.warn(f"ρ estimation failed ({e}); using fallback ρ={fallback}.")
        return fallback


# ---------------------------------------------------------------------------
# SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Dixon-Coles module — sanity checks")
    print("=" * 60)

    # 1. τ values
    lam, mu, rho = 1.2, 0.9, -0.1
    print(f"\nτ values for λ={lam}, μ={mu}, ρ={rho}:")
    for x in range(3):
        for y in range(3):
            print(f"  τ({x},{y}) = {dc_tau(x, y, lam, mu, rho):.4f}")

    # 2. Matrix properties
    for test_rho in [0.0, -0.05, -0.1, -0.15, -0.2]:
        mat = dc_score_matrix(1.5, 1.0, test_rho)
        assert np.all(mat >= 0), f"Negative cell for ρ={test_rho}!"
        assert abs(mat.sum() - 1.0) < 1e-10, f"Sum != 1 for ρ={test_rho}!"
        print(f"  ρ={test_rho:+.2f}  sum={mat.sum():.10f}  min={mat.min():.6f}  ✅")

    # 3. ρ=0 should match independent Poisson
    mat_dc = dc_score_matrix(1.5, 1.0, 0.0)
    h_probs = poisson.pmf(np.arange(9), 1.5)
    a_probs = poisson.pmf(np.arange(9), 1.0)
    mat_ind = np.outer(h_probs, a_probs)
    mat_ind /= mat_ind.sum()
    assert np.allclose(mat_dc, mat_ind, atol=1e-12), "ρ=0 doesn't match independent!"
    print("\n  ρ=0 matches independent Poisson: ✅")

    # 4. Outcome probs
    probs = dc_outcome_probs(1.5, 1.0, -0.1)
    print(f"\n  Outcome probs (λ=1.5, μ=1.0, ρ=−0.1): H={probs[0]:.4f} D={probs[1]:.4f} A={probs[2]:.4f}")
    print(f"  Sum = {probs.sum():.10f}")

    print("\n✅ All sanity checks passed!")
