"""
Genera data-worldcup/squad_values.csv con valores de mercado de plantilla
por selección y año de Mundial, usando datos reales de Transfermarkt via
el dataset de Kaggle davidcariboo/player-scores.

Uso:
    Primera vez:   .venv/bin/python fetch_squad_values.py
    Actualizar:    .venv/bin/python fetch_squad_values.py --force
    Luego:         .venv/bin/python train_model.py   (reentrenar con nuevas features)

Fuente de datos:
    kaggle-football-data-transfermarkt/player_valuations.csv  (historial 2000-hoy)
    kaggle-football-data-transfermarkt/players.csv            (nacionalidad por jugador)
"""

import argparse
import os

import numpy as np
import pandas as pd

from data_prep import TEAM_NAME_MAP

CSV_PATH = "data-worldcup/squad_values.csv"
TM_DIR = "kaggle-football-data-transfermarkt"
SQUAD_SIZE = 26  # FIFA WC squad size (23 pre-2022, 26 since 2022)
WC_YEARS = [1930, 1934, 1938, 1950, 1954, 1958, 1962, 1966,
            1970, 1974, 1978, 1982, 1986, 1990, 1994, 1998,
            2002, 2006, 2010, 2014, 2018, 2022, 2026]

# Transfermarkt uses different country names for some teams — map to our standard
TM_COUNTRY_MAP = {
    "United States":    "USA",
    "Korea, South":     "South Korea",
    "Korea, North":     "North Korea",
    "Ivory Coast":      "Ivory Coast",
    "Congo, DR":        "DR Congo",
    "Republic of Ireland": "Republic of Ireland",
}


def _compute_squad_values_from_tm():
    """Compute per-country squad values for each WC year using Transfermarkt data.

    For each WC year from 2002 to 2026:
      - Filter player_valuations to a 6-month window centred on June of that year.
      - For each player keep the record closest to June 1st of that year.
      - Take the top SQUAD_SIZE players by value per country (proxy for squad selection).
      - Sum those values = squad value for that country in that WC.

    Returns a dict {(team_name, year): squad_value_eur} for years with real data.
    """
    pv_path = os.path.join(TM_DIR, "player_valuations.csv")
    p_path = os.path.join(TM_DIR, "players.csv")

    if not os.path.exists(pv_path) or not os.path.exists(p_path):
        print(f"[AVISO] No se encontraron archivos en {TM_DIR}/. Usando solo fallback estático.")
        return {}

    print("Cargando player_valuations.csv y players.csv...")
    pv = pd.read_csv(pv_path, usecols=["player_id", "date", "market_value_in_eur"])
    players = pd.read_csv(p_path, usecols=["player_id", "country_of_citizenship"])

    pv["date"] = pd.to_datetime(pv["date"])
    pv = pv.dropna(subset=["market_value_in_eur"])
    pv["market_value_in_eur"] = pv["market_value_in_eur"].astype(float)

    # Normalise country names
    players["country_of_citizenship"] = (
        players["country_of_citizenship"]
        .replace(TM_COUNTRY_MAP)
        .replace(TEAM_NAME_MAP)
    )

    pv = pv.merge(players, on="player_id", how="left")
    pv = pv.dropna(subset=["country_of_citizenship"])

    results = {}
    real_years = [y for y in WC_YEARS if y >= 2002]

    for year in real_years:
        target = pd.Timestamp(f"{year}-06-01")
        window_start = target - pd.DateOffset(months=4)
        window_end = target + pd.DateOffset(months=2)

        # Filter to window around tournament start
        window = pv[(pv["date"] >= window_start) & (pv["date"] <= window_end)].copy()

        if len(window) == 0:
            # For 2026, use the most recent available data
            window = pv[pv["date"] <= window_end].copy()
            if len(window) == 0:
                continue

        # Per player, keep the record closest to June 1st
        window["days_diff"] = (window["date"] - target).abs().dt.days
        window = window.sort_values("days_diff")
        window = window.drop_duplicates(subset=["player_id"], keep="first")

        # Per country: take top SQUAD_SIZE players by value, then sum
        squad = (
            window.sort_values("market_value_in_eur", ascending=False)
            .groupby("country_of_citizenship")
            .head(SQUAD_SIZE)
            .groupby("country_of_citizenship")["market_value_in_eur"]
            .sum()
        )

        for country, value in squad.items():
            results[(country, year)] = int(value)

        teams_found = len(squad)
        print(f"  {year}: {teams_found} selecciones con datos reales")

    return results


def fetch_data(force=False):
    if os.path.exists(CSV_PATH) and not force:
        print(f"[{CSV_PATH}] ya existe. Usa --force para sobreescribir.")
        return

    real_data = _compute_squad_values_from_tm()
    print(f"Datos reales obtenidos: {len(real_data)} entradas (selección × año)")

    # Load all (team, year) combos from matches.csv
    from data_prep import load_clean_matches
    df = load_clean_matches()
    df["year"] = df["match_date"].dt.year
    home = df[["home_team_name", "year"]].rename(columns={"home_team_name": "team_name"})
    away = df[["away_team_name", "year"]].rename(columns={"away_team_name": "team_name"})
    all_pairs = pd.concat([home, away]).drop_duplicates()

    # Add 2026 row for every known team (for inference)
    all_teams = all_pairs["team_name"].unique()
    df_2026 = pd.DataFrame({"team_name": all_teams, "year": 2026})
    all_pairs = pd.concat([all_pairs, df_2026]).drop_duplicates()

    # Compute year medians from real data for fallback
    year_totals: dict[int, list] = {}
    for (_, yr), val in real_data.items():
        year_totals.setdefault(yr, []).append(val)
    year_medians = {yr: int(np.median(vals)) for yr, vals in year_totals.items()}

    # Historical medians for years with no TM data (pre-2002)
    # Based on rough historical squad valuations adjusted for era
    HISTORICAL_MEDIANS = {
        1930: 500_000,    1934: 700_000,    1938: 900_000,
        1950: 1_200_000,  1954: 1_500_000,  1958: 2_000_000,
        1962: 2_800_000,  1966: 4_000_000,  1970: 6_000_000,
        1974: 9_000_000,  1978: 12_000_000, 1982: 18_000_000,
        1986: 28_000_000, 1990: 45_000_000, 1994: 70_000_000,
        1998: 110_000_000,
    }

    records = []
    stats = {"real": 0, "fallback": 0}

    for _, row in all_pairs.iterrows():
        team, year = row["team_name"], row["year"]
        if (team, year) in real_data:
            val = real_data[(team, year)]
            stats["real"] += 1
        else:
            # Use year median: real if available, historical estimate otherwise
            median = year_medians.get(year) or HISTORICAL_MEDIANS.get(year, 1_000_000)
            val = median
            stats["fallback"] += 1
        records.append({"team_name": team, "year": int(year), "squad_value_eur": val})

    out = pd.DataFrame(records).sort_values(["year", "team_name"])
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    out.to_csv(CSV_PATH, index=False)

    print(f"\nGuardado en {CSV_PATH}: {len(out)} filas")
    print(f"  Datos reales (TM): {stats['real']}")
    print(f"  Fallback (mediana): {stats['fallback']}")

    # Preview top teams for 2026
    top_2026 = out[out["year"] == 2026].nlargest(10, "squad_value_eur")[["team_name", "squad_value_eur"]]
    top_2026["squad_value_EUR_M"] = (top_2026["squad_value_eur"] / 1e6).round(1)
    print("\nTop 10 selecciones por valor de plantilla (2026):")
    print(top_2026[["team_name", "squad_value_EUR_M"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Sobreescribir CSV si ya existe")
    args = parser.parse_args()
    fetch_data(force=args.force)
