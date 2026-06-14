"""
utils.py

This module handles:
1. Fetching match schedules and team lists from the football-data.org API.
2. Graceful fallback to the offline WC2026 JSON if the API fails.
3. Reading and writing predictions to a JSON file, with Railway Volume support.
"""

import os
import time
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

import json
import requests
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st

from team_names import normalize_team_name, is_placeholder_team

# Global list to track timestamps of football-data.org API requests (to enforce rate limits)
_api_call_timestamps = []

def _rate_limit_check():
    """
    Enforces a maximum of 9 API calls per 60 seconds.
    If the threshold is reached, blocks execution (sleeps) until a slot becomes available.
    """
    global _api_call_timestamps
    now = time.time()
    
    # Keep only timestamps within the last 60 seconds
    _api_call_timestamps = [t for t in _api_call_timestamps if now - t < 60]
    
    if len(_api_call_timestamps) >= 9:
        oldest_call = _api_call_timestamps[0]
        wait_time = 60.0 - (now - oldest_call)
        if wait_time > 0:
            st.warning(f"⏳ Límite de llamadas a la API alcanzado. Esperando {wait_time:.1f} segundos para evitar bloqueo (máx 9/min)...")
            time.sleep(wait_time)
            # Recalculate time after sleep
            now = time.time()
            
    _api_call_timestamps.append(now)

# File database for storing predictions.
# In Railway, mount a Volume at /app/storage. Locally, this falls back to
# predictions.json in the project root unless PREDICTIONS_FILE is set.
PREDICTIONS_FILE = os.environ.get(
    "PREDICTIONS_FILE",
    "/app/storage/predictions.json" if os.path.isdir("/app/storage") else "predictions.json"
)


_SEED_PREDICTIONS_FILE = os.path.join(os.path.dirname(__file__), "predictions.json")


def _ensure_predictions_file():
    """
    Creates the predictions storage directory/file if missing.

    On first run with a fresh Volume, seeds it from the predictions.json
    bundled in the repo so existing predictions aren't lost when the
    storage path changes.
    """
    directory = os.path.dirname(PREDICTIONS_FILE)
    if directory:
        os.makedirs(directory, exist_ok=True)
    if not os.path.exists(PREDICTIONS_FILE):
        seed = []
        if os.path.abspath(PREDICTIONS_FILE) != os.path.abspath(_SEED_PREDICTIONS_FILE) \
                and os.path.exists(_SEED_PREDICTIONS_FILE):
            try:
                with open(_SEED_PREDICTIONS_FILE, "r", encoding="utf-8") as f:
                    seed = json.load(f)
            except (json.JSONDecodeError, IOError):
                seed = []
        with open(PREDICTIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(seed, f, indent=4, ensure_ascii=False)

# ==========================================
# OFFLINE SCHEDULE (WC 2026 JSON)
# ==========================================
# Used as the primary realistic dataset containing the 104 matches of WC 2026
def load_official_wc2026_schedule():
    """
    Reads the downloaded official wc2026_schedule.json and converts it to the 
    football-data.org API standard format expected by the Streamlit application.
    """
    file_path = "data-worldcup/wc2026_schedule.json"
    if not os.path.exists(file_path):
        return []
        
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        matches_list = data.get("matches", [])
        formatted_matches = []
        
        for idx, m in enumerate(matches_list):
            date_str = m.get("date", "")
            # Parse "13:00 UTC-6" into local time + offset, then convert to real UTC
            time_field = m.get("time", "")
            parts = time_field.split(" ")
            time_str = parts[0] if parts else "12:00"
            offset_str = parts[1] if len(parts) > 1 else "UTC+0"
            try:
                offset_hours = int(offset_str.replace("UTC", ""))
            except ValueError:
                offset_hours = 0

            if date_str:
                local_dt = datetime.strptime(f"{date_str}T{time_str}", "%Y-%m-%dT%H:%M")
                utc_dt = local_dt - timedelta(hours=offset_hours)
                utc_date = utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            else:
                utc_date = "2026-06-11T12:00:00Z"
            
            team1 = normalize_team_name(m.get("team1", "TBD"))
            team2 = normalize_team_name(m.get("team2", "TBD"))
            
            match_obj = {
                "id": m.get("num", idx + 1000),  # Use official match number if available
                "status": "SCHEDULED",
                "homeTeam": {"name": team1},
                "awayTeam": {"name": team2},
                "utcDate": utc_date,
                "score": {"fullTime": {"home": None, "away": None}},
                "stage": m.get("round", "Regular")
            }
            formatted_matches.append(match_obj)
            
        return formatted_matches
    except Exception as e:
        st.error(f"Error loading local WC2026 schedule: {e}")
        return []

# ==========================================
# API INTEGRATION (football-data.org)
# ==========================================

@st.cache_data(ttl=600)  # Cache for 10 minutes to prevent API rate limits (10 reqs/min on free tier)
def fetch_wc_matches_api(api_key: str):
    """
    Fetches World Cup matches from football-data.org API.
    If it fails, or if there is no active tournament data, falls back to the downloaded official WC2026 JSON.
    
    :param api_key: Token for authentication.
    :return: (List of matches, boolean indicating whether API was successful)
    """
    offline_matches = load_official_wc2026_schedule()
    
    if not api_key:
        return offline_matches, False
        
    url = "https://api.football-data.org/v4/competitions/WC/matches"
    headers = {"X-Auth-Token": api_key}
    
    try:
        _rate_limit_check()
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            matches = data.get("matches", [])
            # If API succeeds but has no matches (e.g. out of season/not populated), return offline JSON
            if not matches:
                return offline_matches, False
            for m in matches:
                if m.get("homeTeam", {}).get("name"):
                    m["homeTeam"]["name"] = normalize_team_name(m["homeTeam"]["name"])
                if m.get("awayTeam", {}).get("name"):
                    m["awayTeam"]["name"] = normalize_team_name(m["awayTeam"]["name"])
            return matches, True
        else:
            # Fallback to offline JSON
            return offline_matches, False
            
    except requests.exceptions.RequestException as e:
        return offline_matches, False


def get_teams_from_matches(matches):
    """
    Extracts a list of unique team names from matches list.
    """
    teams = set()
    for match in matches:
        home = normalize_team_name(match.get("homeTeam", {}).get("name"))
        away = normalize_team_name(match.get("awayTeam", {}).get("name"))
        if home and not is_placeholder_team(home):
            teams.add(home)
        if away and not is_placeholder_team(away):
            teams.add(away)
    
    return sorted(list(teams))

# ==========================================
# FILE DB MANAGER (predictions.json)
# ==========================================

def load_predictions():
    """
    Reads predictions from the configured JSON file. Creates file if missing.
    """
    try:
        _ensure_predictions_file()
        with open(PREDICTIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        st.error(f"Error loading predictions: {str(e)}. Starting with empty predictions list.")
        return []


def save_prediction(prediction_data):
    """
    Appends a new prediction to the configured JSON file.
    """
    predictions = load_predictions()
    
    # Generate an ID if not present
    if "id" not in prediction_data:
        prediction_data["id"] = datetime.now().strftime("%Y%m%d%H%M%S%f")
        
    prediction_data["timestamp"] = datetime.now().isoformat()
    predictions.append(prediction_data)
    
    try:
        _ensure_predictions_file()
        with open(PREDICTIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=4, ensure_ascii=False)
        return True
    except IOError as e:
        st.error(f"Failed to write prediction to file: {str(e)}")
        return False


def find_prediction(match_id, home_team, away_team):
    """
    Busca una predicción existente para el partido dado.
    - Si match_id es real (no None/"CUSTOM"): busca por match_id (como string).
    - Si no: busca por (home_team, away_team) con match_id == "CUSTOM"/None.
    Retorna el dict de la predicción o None.
    """
    predictions = load_predictions()
    if match_id and match_id != "CUSTOM":
        for p in predictions:
            if str(p.get("match_id")) == str(match_id):
                return p
        return None
    for p in predictions:
        if (p.get("match_id") in (None, "CUSTOM")
                and p.get("home_team") == home_team
                and p.get("away_team") == away_team):
            return p
    return None


def upsert_prediction(prediction_data):
    """
    Si ya existe una predicción para el mismo partido (mismo match_id, o misma
    pareja de equipos cuando match_id es "CUSTOM"), la actualiza in-place
    preservando su "id" original. Si no existe, hace append como una nueva.
    Retorna (success, was_update).
    """
    predictions = load_predictions()

    existing = find_prediction(
        prediction_data.get("match_id"),
        prediction_data.get("home_team"),
        prediction_data.get("away_team"),
    )

    prediction_data["timestamp"] = datetime.now().isoformat()

    if existing:
        prediction_data["id"] = existing["id"]
        for i, p in enumerate(predictions):
            if p.get("id") == existing["id"]:
                predictions[i] = prediction_data
                break
    else:
        if "id" not in prediction_data:
            prediction_data["id"] = datetime.now().strftime("%Y%m%d%H%M%S%f")
        predictions.append(prediction_data)

    try:
        _ensure_predictions_file()
        with open(PREDICTIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=4, ensure_ascii=False)
        return True, existing is not None
    except IOError as e:
        st.error(f"Failed to write prediction to file: {str(e)}")
        return False, False


def delete_prediction(prediction_id):
    """
    Deletes a prediction from the configured JSON file by ID.
    """
    predictions = load_predictions()
    updated_predictions = [p for p in predictions if p.get("id") != prediction_id]
    
    try:
        _ensure_predictions_file()
        with open(PREDICTIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(updated_predictions, f, indent=4, ensure_ascii=False)
        return True
    except IOError as e:
        st.error(f"Failed to delete prediction: {str(e)}")
        return False
