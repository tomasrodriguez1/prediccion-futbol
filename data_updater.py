"""
data_updater.py

Pure (Streamlit-free) logic to grow the training corpus with new results:

  1. add_match()         — append a single manually-entered match.
  2. fetch_api_results() — pull FINISHED World Cup matches from football-data.org.
  3. retrain()           — retrain the RF models on the updated corpus.
  4. corpus_status()     — quick health snapshot for the UI.

All new rows land in data-worldcup/extra_matches.csv (same raw schema as
all_matches.csv) and are deduplicated against the historical data by
(date, home, away) inside data_prep.load_combined_corpus().
"""

import os
from datetime import datetime, date

import pandas as pd
import requests

from data_prep import EXTRA_MATCHES_CSV
from team_names import normalize_team_name

EXTRA_COLUMNS = ["date", "home_team", "away_team",
                 "home_score", "away_score", "tournament", "country"]

FOOTBALL_DATA_URL = "https://api.football-data.org/v4/competitions/WC/matches"


# ---------------------------------------------------------------------------
# LOW-LEVEL CSV HELPERS
# ---------------------------------------------------------------------------

def _read_extra() -> pd.DataFrame:
    """Read extra_matches.csv (creating an empty, typed frame if missing)."""
    if os.path.exists(EXTRA_MATCHES_CSV):
        df = pd.read_csv(EXTRA_MATCHES_CSV)
        if not df.empty:
            return df
    return pd.DataFrame(columns=EXTRA_COLUMNS)


def _write_extra(df: pd.DataFrame) -> None:
    df = df[EXTRA_COLUMNS].sort_values("date").reset_index(drop=True)
    df.to_csv(EXTRA_MATCHES_CSV, index=False)


def _existing_keys() -> set[tuple]:
    """Set of (date, normalized_home, normalized_away) across history + extra.

    Used to avoid inserting duplicates of matches already in the corpus.
    """
    from data_prep import load_combined_corpus
    corpus = load_combined_corpus()
    keys = set(
        zip(
            corpus["match_date"].dt.strftime("%Y-%m-%d"),
            corpus["home_team_name"],
            corpus["away_team_name"],
        )
    )
    return keys


# ---------------------------------------------------------------------------
# 1. MANUAL ENTRY
# ---------------------------------------------------------------------------

def add_match(match_date, home_team: str, away_team: str,
              home_score: int, away_score: int,
              tournament: str = "Friendly", country: str = "") -> tuple[bool, str]:
    """Append a single match to extra_matches.csv.

    Returns (ok, message). Rejects malformed input and duplicates.
    """
    # --- Validation -------------------------------------------------------
    if isinstance(match_date, (datetime, date)):
        date_str = match_date.strftime("%Y-%m-%d")
    else:
        try:
            date_str = pd.to_datetime(match_date).strftime("%Y-%m-%d")
        except Exception:
            return False, f"Fecha inválida: {match_date!r} (usa YYYY-MM-DD)."

    home_team = (home_team or "").strip()
    away_team = (away_team or "").strip()
    if not home_team or not away_team:
        return False, "Debes indicar ambos equipos."
    if home_team == away_team:
        return False, "Un equipo no puede jugar contra sí mismo."

    try:
        home_score = int(home_score)
        away_score = int(away_score)
    except (TypeError, ValueError):
        return False, "Los goles deben ser números enteros."
    if home_score < 0 or away_score < 0:
        return False, "Los goles no pueden ser negativos."

    tournament = (tournament or "Friendly").strip()
    country = (country or "").strip()

    # --- Duplicate check (against normalized names) -----------------------
    norm_home = normalize_team_name(home_team)
    norm_away = normalize_team_name(away_team)
    if (date_str, norm_home, norm_away) in _existing_keys():
        return False, f"Ese partido ya existe en el corpus ({date_str} {norm_home} vs {norm_away})."

    # --- Append -----------------------------------------------------------
    df = _read_extra()
    new_row = {
        "date": date_str,
        "home_team": home_team,
        "away_team": away_team,
        "home_score": home_score,
        "away_score": away_score,
        "tournament": tournament,
        "country": country,
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    _write_extra(df)
    return True, f"Añadido: {date_str} {home_team} {home_score}-{away_score} {away_team} ({tournament})."


# ---------------------------------------------------------------------------
# 2. API SYNC (football-data.org)
# ---------------------------------------------------------------------------

def fetch_api_results(api_key: str, timeout: int = 15) -> tuple[bool, str, int]:
    """Pull FINISHED World Cup matches and append the new ones.

    Returns (ok, message, n_added).
    """
    if not api_key:
        return False, "No hay API key configurada.", 0

    try:
        resp = requests.get(FOOTBALL_DATA_URL,
                            headers={"X-Auth-Token": api_key}, timeout=timeout)
    except requests.exceptions.RequestException as e:
        return False, f"Error de red al consultar la API: {e}", 0

    if resp.status_code != 200:
        return False, f"La API respondió {resp.status_code}.", 0

    matches = resp.json().get("matches", [])
    finished = [m for m in matches if m.get("status") == "FINISHED"]
    if not finished:
        return True, "La API no devolvió partidos finalizados todavía.", 0

    existing = _existing_keys()
    df = _read_extra()
    extra_keys = set(
        zip(df["date"].astype(str),
            df["home_team"].map(normalize_team_name),
            df["away_team"].map(normalize_team_name))
    ) if not df.empty else set()

    new_rows = []
    for m in finished:
        utc = m.get("utcDate", "")
        date_str = utc.split("T")[0] if utc else None
        home = m.get("homeTeam", {}).get("name")
        away = m.get("awayTeam", {}).get("name")
        full = m.get("score", {}).get("fullTime", {})
        hs, as_ = full.get("home"), full.get("away")
        if not (date_str and home and away) or hs is None or as_ is None:
            continue

        norm_home = normalize_team_name(home)
        norm_away = normalize_team_name(away)
        key = (date_str, norm_home, norm_away)
        if key in existing or key in extra_keys:
            continue

        new_rows.append({
            "date": date_str,
            "home_team": home,
            "away_team": away,
            "home_score": int(hs),
            "away_score": int(as_),
            "tournament": "World Cup",
            "country": "",
        })
        extra_keys.add(key)

    if not new_rows:
        return True, "Sin resultados nuevos: el corpus ya está al día.", 0

    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    _write_extra(df)
    return True, f"Añadidos {len(new_rows)} partidos nuevos del Mundial.", len(new_rows)


# ---------------------------------------------------------------------------
# 3. RETRAIN
# ---------------------------------------------------------------------------

def retrain() -> tuple[bool, str]:
    """Retrain the RF models on the updated corpus."""
    try:
        from train_model import train_models
        train_models()
        return True, "Modelos reentrenados con los datos actualizados."
    except Exception as e:
        return False, f"Error al reentrenar: {e}"


# ---------------------------------------------------------------------------
# 4. STATUS
# ---------------------------------------------------------------------------

def corpus_status() -> dict:
    """Return a snapshot for the UI: last match, extra count, last train time."""
    from data_prep import load_combined_corpus
    corpus = load_combined_corpus()
    extra = _read_extra()

    model_path = "models/home_model.pkl"
    last_train = (
        datetime.fromtimestamp(os.path.getmtime(model_path)).strftime("%Y-%m-%d %H:%M")
        if os.path.exists(model_path) else "nunca"
    )

    return {
        "total_matches": len(corpus),
        "last_match_date": corpus["match_date"].max().strftime("%Y-%m-%d"),
        "extra_matches": len(extra),
        "last_train": last_train,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Actualizar corpus de partidos.")
    parser.add_argument("--add", metavar="FECHA,LOCAL,VISITA,GL,GV,TORNEO,PAIS",
                        help="Añadir un partido manual (campos separados por coma).")
    parser.add_argument("--fetch-api", action="store_true",
                        help="Sincronizar resultados FINISHED del Mundial vía API.")
    parser.add_argument("--retrain", action="store_true",
                        help="Reentrenar los modelos al final.")
    parser.add_argument("--status", action="store_true", help="Mostrar estado del corpus.")
    args = parser.parse_args()

    if args.status:
        print(corpus_status())

    if args.add:
        parts = [p.strip() for p in args.add.split(",")]
        if len(parts) < 5:
            print("Formato: FECHA,LOCAL,VISITA,GL,GV[,TORNEO,PAIS]")
        else:
            fecha, local, visita, gl, gv = parts[:5]
            torneo = parts[5] if len(parts) > 5 else "Friendly"
            pais = parts[6] if len(parts) > 6 else ""
            ok, msg = add_match(fecha, local, visita, gl, gv, torneo, pais)
            print(("OK: " if ok else "ERROR: ") + msg)

    if args.fetch_api:
        key = os.environ.get("FOOTBALL_DATA_API_KEY", "")
        ok, msg, n = fetch_api_results(key)
        print(("OK: " if ok else "ERROR: ") + msg)

    if args.retrain:
        ok, msg = retrain()
        print(("OK: " if ok else "ERROR: ") + msg)
