"""
app.py

This is the main entry point for the Polla del Mundial 2026 Streamlit web application.
It integrates:
- Premium frontend aesthetics with glassmorphic cards and CSS overrides.
- Sidebar configuration for API Keys (football-data.org).
- Tabbed layout: Predictor Hub, Saved Predictions Dashboard, Team Ratings & Stats, and Tournament Fixtures.
- Interactive Plotly visualizations for outcomes and score distribution heatmaps.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from html import escape

# Import modules from local files
from utils import (
    fetch_wc_matches_api,
    get_teams_from_matches,
    load_predictions,
    save_prediction,
    delete_prediction,
    PREDICTIONS_FILE
)
from predictor import calculate_team_ratings, predict_match, SEED_RATINGS, DEFAULT_TOURNAMENT_AVG
from simulation import run_monte_carlo, WC2026_GROUPS, ALL_TEAMS
from team_names import normalize_team_name
import data_updater

# ==========================================
# PAGE CONFIG & CSS INJECTION
# ==========================================
st.set_page_config(
    page_title="Polla Mundial 2026 - Poisson Predictor",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def inject_premium_css():
    """
    Injects custom styles to give the app a premium, high-tech glassmorphism theme, 
    matching modern web design standards.
    """
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
            
            /* Apply custom font globally */
            html, body, [class*="css"] {
                font-family: 'Outfit', sans-serif;
            }

            html, body, .stApp, [data-testid="stAppViewContainer"] {
                background:
                    radial-gradient(circle at top left, rgba(0, 230, 118, 0.10), transparent 28rem),
                    linear-gradient(180deg, #0d111d 0%, #111827 48%, #0d111d 100%) !important;
                color: #eef4ff !important;
            }

            [data-testid="stHeader"] {
                background: transparent !important;
            }

            [data-testid="stMarkdownContainer"],
            [data-testid="stMarkdownContainer"] p,
            [data-testid="stMarkdownContainer"] li,
            label,
            .stSelectbox label,
            .stMultiSelect label,
            .stRadio label,
            .stSlider label,
            .stNumberInput label,
            .stTextInput label {
                color: #dbe7ff;
            }

            .main .block-container {
                max-width: 1280px;
                padding-top: 2rem;
                padding-left: 2rem;
                padding-right: 2rem;
            }
            
            /* Custom glassmorphism card styling */
            .glass-card {
                background: rgba(23, 28, 41, 0.6) !important;
                backdrop-filter: blur(12px) !important;
                -webkit-backdrop-filter: blur(12px) !important;
                border: 1px solid rgba(255, 255, 255, 0.08) !important;
                border-radius: 16px !important;
                padding: 24px;
                margin-bottom: 20px;
                box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.35);
                transition: transform 0.2s ease, border-color 0.2s ease;
            }
            .glass-card:empty {
                display: none;
            }
            .glass-card:hover {
                transform: translateY(-2px);
                border-color: rgba(0, 230, 118, 0.4) !important;
            }
            
            /* Gradient highlights */
            .gradient-title {
                background: linear-gradient(135deg, #00E676 0%, #00B0FF 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-weight: 700;
                font-size: 2.2rem;
                margin-bottom: 5px;
            }
            
            .gradient-subtitle {
                background: linear-gradient(135deg, #00B0FF 0%, #D500F9 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-weight: 600;
                font-size: 1.3rem;
            }
            
            /* Custom labels and status badges */
            .badge {
                display: inline-block;
                padding: 4px 10px;
                border-radius: 20px;
                font-size: 0.75rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            .badge-finished {
                background-color: rgba(0, 230, 118, 0.12);
                color: #00E676;
                border: 1px solid rgba(0, 230, 118, 0.3);
            }
            .badge-scheduled {
                background-color: rgba(0, 176, 255, 0.12);
                color: #00B0FF;
                border: 1px solid rgba(0, 176, 255, 0.3);
            }
            .badge-api {
                background-color: rgba(0, 230, 118, 0.15);
                color: #00E676;
                font-size: 0.8rem;
                border: 1px solid rgba(0, 230, 118, 0.4);
            }
            .badge-mock {
                background-color: rgba(255, 145, 0, 0.15);
                color: #FF9100;
                font-size: 0.8rem;
                border: 1px solid rgba(255, 145, 0, 0.4);
            }
            
            /* Styled submit buttons */
            div.stButton > button {
                background: linear-gradient(135deg, #00E676 0%, #00B0FF 100%) !important;
                color: white !important;
                border: none !important;
                border-radius: 12px !important;
                padding: 12px 28px !important;
                font-weight: 600 !important;
                font-size: 1.05rem !important;
                width: 100%;
                box-shadow: 0 4px 15px rgba(0, 176, 255, 0.35) !important;
                transition: all 0.3s ease !important;
            }
            div.stButton > button:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 6px 20px rgba(0, 230, 118, 0.5) !important;
            }
            
            /* Styled Delete buttons */
            div.stButton > button.delete-btn {
                background: rgba(255, 23, 68, 0.15) !important;
                color: #FF1744 !important;
                border: 1px solid rgba(255, 23, 68, 0.3) !important;
                padding: 4px 10px !important;
                font-size: 0.8rem !important;
                box-shadow: none !important;
            }
            div.stButton > button.delete-btn:hover {
                background: #FF1744 !important;
                color: white !important;
                transform: none !important;
            }
            
            /* Sidebar customizations */
            section[data-testid="stSidebar"] {
                background-color: #0d111d !important;
                border-right: 1px solid rgba(255, 255, 255, 0.05);
            }
            
            /* Tabs custom styling */
            .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
                overflow-x: auto;
                scrollbar-width: thin;
            }
            .stTabs [data-baseweb="tab"] {
                height: 45px;
                flex: 0 0 auto;
                background-color: rgba(23, 28, 41, 0.4);
                border-radius: 8px 8px 0px 0px;
                border: 1px solid rgba(255, 255, 255, 0.05);
                border-bottom: none;
                padding: 10px 20px;
                color: #8892b0;
                font-weight: 600;
            }
            .stTabs [aria-selected="true"] {
                background-color: rgba(23, 28, 41, 0.95) !important;
                color: #00E676 !important;
                border-top: 2px solid #00E676 !important;
            }

            .score-pick-card {
                background: rgba(255,255,255,0.03);
                border: 1px solid rgba(255,255,255,0.05);
                padding: 10px;
                border-radius: 10px;
                text-align: center;
            }

            @media (max-width: 768px) {
                .main .block-container {
                    padding: 0.85rem 0.75rem 2rem;
                    max-width: 100%;
                }

                .glass-card {
                    padding: 14px;
                    margin-bottom: 12px;
                    border-radius: 10px !important;
                    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.28);
                }

                .glass-card:hover {
                    transform: none;
                }

                .gradient-title {
                    font-size: 1.75rem !important;
                    line-height: 1.1;
                }

                h1, h2, h3 {
                    line-height: 1.15 !important;
                }

                h3 {
                    font-size: 1.15rem !important;
                }

                h4 {
                    font-size: 1rem !important;
                }

                p, li, label, [data-testid="stMarkdownContainer"] {
                    font-size: 0.93rem;
                }

                .stTabs [data-baseweb="tab-list"] {
                    gap: 6px;
                    padding-bottom: 4px;
                    flex-wrap: nowrap;
                }

                .stTabs [data-baseweb="tab"] {
                    height: 38px;
                    padding: 8px 12px;
                    font-size: 0.82rem;
                    border-radius: 7px;
                    white-space: nowrap;
                }

                div[data-testid="stHorizontalBlock"] {
                    gap: 0.75rem;
                    flex-wrap: wrap;
                }

                div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
                    min-width: 100% !important;
                    width: 100% !important;
                    flex: 1 1 100% !important;
                }

                div[data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > div[data-testid="column"] {
                    min-width: calc(50% - 0.4rem) !important;
                    width: calc(50% - 0.4rem) !important;
                    flex: 1 1 calc(50% - 0.4rem) !important;
                }

                [data-testid="stMetric"] {
                    background: rgba(255, 255, 255, 0.025);
                    border: 1px solid rgba(255, 255, 255, 0.06);
                    border-radius: 10px;
                    padding: 10px;
                }

                [data-testid="stMetricLabel"] p {
                    font-size: 0.78rem;
                }

                [data-testid="stMetricValue"] {
                    font-size: 1.35rem;
                }

                div.stButton > button {
                    padding: 10px 16px !important;
                    font-size: 0.95rem !important;
                    min-height: 42px;
                    transform: none !important;
                }

                div.stButton > button:hover {
                    transform: none !important;
                }

                .stDataFrame {
                    font-size: 0.82rem;
                }

                div[data-testid="stExpander"] details {
                    border-radius: 10px;
                    border-color: rgba(255,255,255,0.08);
                    background: rgba(255,255,255,0.02);
                }
            }
        </style>
        """,
        unsafe_allow_html=True
    )


def inject_pwa_metadata():
    """
    Adds PWA metadata and registers the static service worker from the browser.
    Streamlit renders this through an iframe component, so the script writes tags
    into the parent document head at runtime.
    """
    components.html(
        """
        <script>
        (() => {
            const appTitle = "Polla Mundial 2026";
            const themeColor = "#0d111d";
            const parentDocument = window.parent?.document || document;
            const parentNavigator = window.parent?.navigator || navigator;

            parentDocument.title = appTitle;

            const ensureTag = (selector, createTag) => {
                let tag = parentDocument.head.querySelector(selector);
                if (!tag) {
                    tag = createTag();
                    parentDocument.head.appendChild(tag);
                }
                return tag;
            };

            const manifest = ensureTag('link[rel="manifest"]', () => {
                const link = parentDocument.createElement("link");
                link.rel = "manifest";
                return link;
            });
            manifest.href = "/app/static/manifest.json";

            const appleIcon = ensureTag('link[rel="apple-touch-icon"]', () => {
                const link = parentDocument.createElement("link");
                link.rel = "apple-touch-icon";
                return link;
            });
            appleIcon.href = "/app/static/icon-192.png";

            const metaValues = {
                "theme-color": themeColor,
                "apple-mobile-web-app-capable": "yes",
                "apple-mobile-web-app-status-bar-style": "black-translucent",
                "apple-mobile-web-app-title": appTitle,
                "mobile-web-app-capable": "yes",
                "application-name": appTitle,
            };

            Object.entries(metaValues).forEach(([name, content]) => {
                const meta = ensureTag(`meta[name="${name}"]`, () => {
                    const tag = parentDocument.createElement("meta");
                    tag.name = name;
                    return tag;
                });
                meta.content = content;
            });

            if ("serviceWorker" in parentNavigator) {
                parentNavigator.serviceWorker
                    .register("/app/static/sw.js", { scope: "/app/static/" })
                    .catch(() => {});
            }
        })();
        </script>
        """,
        height=0,
        width=0,
    )


PLOTLY_TRANSPARENT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#ffffff", size=12),
)

PLOTLY_CONFIG = {
    "displayModeBar": False,
    "responsive": True,
}

inject_premium_css()
inject_pwa_metadata()

# ==========================================
# SIDEBAR CONFIGURATION (SETTINGS)
# ==========================================
st.sidebar.markdown('<div style="text-align: center;"><h2 class="gradient-title" style="font-size:1.8rem; margin-top:0;">🏆 Configuración</h2></div>', unsafe_allow_html=True)
st.sidebar.markdown("Configure sus credenciales de API para conectarse a datos en vivo (Opcional).")

# API Key input field
api_key = st.sidebar.text_input(
    "🔑 football-data.org API Key",
    type="password",
    help="Obtenga una API key gratuita registrándose en football-data.org.",
    value=os.environ.get("FOOTBALL_DATA_API_KEY", "")
)

st.sidebar.divider()

# ==========================================
# FETCH DATA & COMPUTE RATINGS
# ==========================================
# Load competition data (with visual load feedback)
matches, is_api_success = fetch_wc_matches_api(api_key)

# API vs Fallback Mode indicator
if is_api_success:
    st.sidebar.markdown('<div class="badge badge-api" style="width: 100%; text-align: center;">✅ API Conectada (En Vivo)</div>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('<div class="badge badge-api" style="width: 100%; text-align: center;">✅ Calendario Offline (JSON)</div>', unsafe_allow_html=True)

# Clear Cache Helper
if st.sidebar.button("🔄 Actualizar Datos (Limpiar Caché)"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown(
    """
    <div style="font-size:0.8rem; color:#8892b0; margin-top:20px; text-align:center;">
        Polla Mundial 2026 v1.0.0<br>
        Implementado con Inteligencia Artificial
    </div>
    """, 
    unsafe_allow_html=True
)

finished_matches = [m for m in matches if m.get("status") == "FINISHED"]
team_ratings = calculate_team_ratings(finished_matches)
available_teams = get_teams_from_matches(matches)

playable_teams = [normalize_team_name(t) for t in available_teams]
playable_teams = sorted(set(t for t in playable_teams if t))
if not playable_teams:
    playable_teams = ["Mexico", "Canada", "USA"] # Fallback if empty


def _outcome_label(home_goals, away_goals):
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def _build_real_results_lookup(matches):
    """Build lookup tables for completed results from API/offline data and manual entries."""
    by_id = {}
    by_teams = {}

    def add_result(match_id, home, away, home_score, away_score, status="FINISHED", source="Calendario"):
        if home_score is None or away_score is None:
            return

        home_norm = normalize_team_name(home)
        away_norm = normalize_team_name(away)
        if not home_norm or not away_norm:
            return

        result = {
            "match_id": str(match_id) if match_id is not None else None,
            "home_team": home_norm,
            "away_team": away_norm,
            "home_score": int(home_score),
            "away_score": int(away_score),
            "status": status,
            "source": source,
        }

        if result["match_id"]:
            by_id[result["match_id"]] = result
        by_teams[(home_norm, away_norm)] = result

    for match in matches:
        full_time = match.get("score", {}).get("fullTime", {})
        status = match.get("status", "SCHEDULED")
        if status == "FINISHED":
            add_result(
                match.get("id"),
                match.get("homeTeam", {}).get("name"),
                match.get("awayTeam", {}).get("name"),
                full_time.get("home"),
                full_time.get("away"),
                status=status,
                source="API" if is_api_success else "Calendario",
            )

    try:
        extra_results = data_updater._read_extra()
        for _, row in extra_results.iterrows():
            add_result(
                None,
                row.get("home_team"),
                row.get("away_team"),
                row.get("home_score"),
                row.get("away_score"),
                status="FINISHED",
                source=row.get("tournament") or "Manual",
            )
    except Exception:
        pass

    return by_id, by_teams


def _real_result_for_prediction(prediction, results_by_id, results_by_teams):
    match_id = str(prediction.get("match_id", ""))
    if match_id in results_by_id:
        return results_by_id[match_id]

    home = normalize_team_name(prediction.get("home_team"))
    away = normalize_team_name(prediction.get("away_team"))
    return results_by_teams.get((home, away))


def _parse_match_datetime(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def _build_match_schedule_lookup(matches):
    by_id = {}
    by_teams = {}

    for match in matches:
        home = normalize_team_name(match.get("homeTeam", {}).get("name"))
        away = normalize_team_name(match.get("awayTeam", {}).get("name"))
        if not home or not away:
            continue

        scheduled_at = _parse_match_datetime(match.get("utcDate"))
        schedule_info = {
            "match_id": str(match.get("id")) if match.get("id") is not None else None,
            "home_team": home,
            "away_team": away,
            "scheduled_at": scheduled_at,
            "status": match.get("status", "SCHEDULED"),
            "stage": match.get("stage", ""),
        }

        if schedule_info["match_id"]:
            by_id[schedule_info["match_id"]] = schedule_info
        by_teams[(home, away)] = schedule_info

    return by_id, by_teams


def _schedule_for_prediction(prediction, schedule_by_id, schedule_by_teams):
    match_id = str(prediction.get("match_id", ""))
    if match_id in schedule_by_id:
        return schedule_by_id[match_id]

    home = normalize_team_name(prediction.get("home_team"))
    away = normalize_team_name(prediction.get("away_team"))
    return schedule_by_teams.get((home, away))


def _format_match_datetime(value):
    if not value:
        return "Fecha por confirmar"
    return value.strftime("%d/%m/%Y %H:%M")


def _evaluate_prediction(prediction, real_result):
    base = {
        "category": "pending",
        "label": "Pendiente",
        "points": 0,
        "predicted_goal_diff": None,
        "real_goal_diff": None,
        "color": "#8892b0",
        "background": "rgba(136, 146, 176, 0.12)",
        "border": "rgba(136, 146, 176, 0.25)",
    }

    if not real_result:
        return base

    pred_home = int(prediction["predicted_goals_home"])
    pred_away = int(prediction["predicted_goals_away"])
    real_home = int(real_result["home_score"])
    real_away = int(real_result["away_score"])
    predicted_goal_diff = pred_home - pred_away
    real_goal_diff = real_home - real_away

    base.update({
        "predicted_goal_diff": predicted_goal_diff,
        "real_goal_diff": real_goal_diff,
    })

    if pred_home == real_home and pred_away == real_away:
        return {
            **base,
            "category": "exact",
            "label": "Marcador exacto",
            "points": 5,
            "color": "#00E676",
            "background": "rgba(0, 230, 118, 0.12)",
            "border": "rgba(0, 230, 118, 0.35)",
        }

    if predicted_goal_diff == real_goal_diff:
        return {
            **base,
            "category": "diff",
            "label": "Diferencia acertada",
            "points": 3,
            "color": "#00B0FF",
            "background": "rgba(0, 176, 255, 0.12)",
            "border": "rgba(0, 176, 255, 0.35)",
        }

    if _outcome_label(pred_home, pred_away) == _outcome_label(real_home, real_away):
        return {
            **base,
            "category": "winner",
            "label": "Ganador acertado",
            "points": 2,
            "color": "#FFD54F",
            "background": "rgba(255, 213, 79, 0.12)",
            "border": "rgba(255, 213, 79, 0.35)",
        }

    return {
        **base,
        "category": "miss",
        "label": "No acertada",
        "color": "#FF5252",
        "background": "rgba(255, 82, 82, 0.12)",
        "border": "rgba(255, 82, 82, 0.35)",
    }

# ==========================================
# MAIN APPLICATION BODY
# ==========================================
st.markdown('<h1 class="gradient-title" style="margin-bottom:0;">🏆 Polla Mundial 2026</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#8892b0; font-size:1.1rem; margin-top:0;">Lógica Híbrida: Random Forest + Distribución de Poisson</p>', unsafe_allow_html=True)

# Define application tabs
tab_dashboard, tab_predictor, tab_ratings, tab_fixtures, tab_sim, tab_data = st.tabs([
    "📊 Predicciones",
    "🎯 Pronóstico",
    "📈 Ratings",
    "📅 Calendario",
    "🎲 Simular",
    "🔄 Datos"
])

# ------------------------------------------
# TAB 1: HUB DE PREDICCIÓN
# ------------------------------------------
with tab_predictor:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h3>Configure su Pronóstico</h3>', unsafe_allow_html=True)
    
    # 1. Selection Modes: Scheduled matches vs Custom matchup
    matchup_mode = st.radio(
        "Método de Selección de Partido:",
        ["Seleccionar del calendario oficial de la Copa del Mundo 2026", "Crear enfrentamiento personalizado"],
        horizontal=True
    )
    
    selected_match_id = None
    
    if matchup_mode == "Seleccionar del calendario oficial de la Copa del Mundo 2026":
        # Filter upcoming scheduled/timed matches
        upcoming_matches = [m for m in matches if m.get("status") in ["SCHEDULED", "TIMED", "LIVE", "IN_PLAY"]]
        
        if upcoming_matches:
            match_options = []
            match_mapping = {}
            for m in upcoming_matches:
                home = m["homeTeam"]["name"]
                away = m["awayTeam"]["name"]
                date_str = m.get("utcDate", "")
                if date_str:
                    dt = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
                    formatted_date = dt.strftime("%d %b - %H:%M")
                else:
                    formatted_date = "TBD"
                
                label = f"⚽ {home} vs {away} ({formatted_date})"
                match_options.append(label)
                match_mapping[label] = m
                
            selected_label = st.selectbox("Seleccione el partido a pronosticar:", match_options)
            chosen_match = match_mapping[selected_label]
            home_team = normalize_team_name(chosen_match["homeTeam"]["name"])
            away_team = normalize_team_name(chosen_match["awayTeam"]["name"])
            selected_match_id = chosen_match["id"]
        else:
            st.info("No hay partidos programados pendientes en la API. Cree un enfrentamiento personalizado en su lugar.")
            home_team = playable_teams[0] if len(playable_teams) > 0 else "Argentina"
            away_team = playable_teams[1] if len(playable_teams) > 1 else "France"
    else:
        # Custom Matchup selector
        col_sel_home, col_sel_away = st.columns(2)
        with col_sel_home:
            home_team = st.selectbox("Equipo Local (Home):", playable_teams, index=0)
        with col_sel_away:
            # Filter home team to avoid playing against itself
            away_options = [t for t in playable_teams if t != home_team]
            away_team = st.selectbox("Equipo Visitante (Away):", away_options, index=0)
            
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 2. Advanced rating controls are collapsed by default so mobile users see
    # the probabilities quickly after choosing a match.
    with st.expander("Ajustes avanzados de fuerza de equipos", expanded=False):
        st.markdown('<div class="glass-card" style="height:100%;">', unsafe_allow_html=True)
        st.markdown('<h4>📊 Modificar Fuerza de Ataque/Defensa</h4>', unsafe_allow_html=True)
        st.markdown(
            """
            <p style="font-size:0.85rem; color:#8892b0; margin-top:0;">
                Los sliders cargan automáticamente los ratings calculados por el modelo de Poisson basándose en los datos de la API.
                Puede ajustarlos si cree que un equipo jugará mejor/peor que su historial de datos.
            </p>
            """, 
            unsafe_allow_html=True
        )
        
        # Load computed strengths
        home_rating_data = team_ratings.get(home_team, SEED_RATINGS.get(home_team, {"attack": 1.0, "defense": 1.0}))
        away_rating_data = team_ratings.get(away_team, SEED_RATINGS.get(away_team, {"attack": 1.0, "defense": 1.0}))
        
        st.markdown(f"**🔥 {home_team}**")
        home_attack = st.slider(
            "Fuerza de Ataque (Local)", 
            min_value=0.2, max_value=2.5, 
            value=float(home_rating_data["attack"]), step=0.05, key="h_att"
        )
        home_defense = st.slider(
            "Fuerza de Defensa (Local - Menos es mejor)", 
            min_value=0.2, max_value=2.5, 
            value=float(home_rating_data["defense"]), step=0.05, key="h_def"
        )
        
        st.divider()
        
        st.markdown(f"**✈️ {away_team}**")
        away_attack = st.slider(
            "Fuerza de Ataque (Visitante)", 
            min_value=0.2, max_value=2.5, 
            value=float(away_rating_data["attack"]), step=0.05, key="a_att"
        )
        away_defense = st.slider(
            "Fuerza de Defensa (Visitante - Menos es mejor)", 
            min_value=0.2, max_value=2.5, 
            value=float(away_rating_data["defense"]), step=0.05, key="a_def"
        )
        
        # Compute prediction first to get the correct lambdas (AI or Slider-based)
        custom_ratings = {
            home_team: {"attack": home_attack, "defense": home_defense},
            away_team: {"attack": away_attack, "defense": away_defense}
        }
        
        home_team = normalize_team_name(home_team)
        away_team = normalize_team_name(away_team)
        prediction = predict_match(home_team, away_team, custom_ratings, tournament_avg=DEFAULT_TOURNAMENT_AVG)
        
        st.markdown("---")
        if prediction.get("ml_used", False):
            st.markdown("**🤖 Goles Esperados (xG) por Machine Learning:**")
            st.caption(
                "ℹ️ Con el modelo de IA activo, el xG se calcula a partir de Elo, "
                "estadísticas históricas y ventaja de anfitrión. Los sliders de "
                "arriba solo se aplican en el modo Poisson de respaldo (sin modelos entrenados)."
            )
        else:
            st.markdown("**Goles Esperados Basados en Poisson (λ):**")
            
        col_lambda_h, col_lambda_a = st.columns(2)
        col_lambda_h.metric(f"xG {home_team}", f"{prediction['expected_home_goals']:.2f} goles")
        col_lambda_a.metric(f"xG {away_team}", f"{prediction['expected_away_goals']:.2f} goles")
        st.markdown('</div>', unsafe_allow_html=True)
        
    analytics_col, heatmap_col = st.columns([1, 1])

    with analytics_col:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        if prediction.get("ml_used", False):
            st.markdown('<h4>📈 Probabilidades (IA Híbrida)</h4>', unsafe_allow_html=True)
            st.markdown('<div class="badge badge-api" style="margin-bottom: 15px;">🤖 Powered by Random Forest AI</div>', unsafe_allow_html=True)
        else:
            st.markdown('<h4>📈 Probabilidades (Modelo Poisson)</h4>', unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=['Resultado'],
            x=[prediction["home_win_prob"] * 100],
            name=f"{home_team}",
            orientation='h',
            marker=dict(color='#00E676', line=dict(color='rgba(0,0,0,0)', width=0))
        ))
        fig.add_trace(go.Bar(
            y=['Resultado'],
            x=[prediction["draw_prob"] * 100],
            name="Empate",
            orientation='h',
            marker=dict(color='#757575')
        ))
        fig.add_trace(go.Bar(
            y=['Resultado'],
            x=[prediction["away_win_prob"] * 100],
            name=f"{away_team}",
            orientation='h',
            marker=dict(color='#00B0FF')
        ))
        fig.update_layout(
            **PLOTLY_TRANSPARENT_LAYOUT,
            barmode='stack',
            height=120,
            margin=dict(l=6, r=6, t=8, b=8),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=10)),
            xaxis=dict(showgrid=False, zeroline=False, range=[0, 100], showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

        col_pct_h, col_pct_d, col_pct_a = st.columns(3)
        col_pct_h.metric(home_team, f"{prediction['home_win_prob']*100:.1f}%")
        col_pct_d.metric("Empate", f"{prediction['draw_prob']*100:.1f}%")
        col_pct_a.metric(away_team, f"{prediction['away_win_prob']*100:.1f}%")

        st.markdown('</div>', unsafe_allow_html=True)

    with heatmap_col:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<h4>🔥 Matriz de Marcador</h4>', unsafe_allow_html=True)
        matrix = prediction["score_matrix"][:6, :6]  # 0 to 5 goals

        fig_heatmap = px.imshow(
            matrix * 100,
            labels=dict(x=f"Goles {away_team}", y=f"Goles {home_team}", color="Probabilidad (%)"),
            x=[str(i) for i in range(6)],
            y=[str(i) for i in range(6)],
            color_continuous_scale="magma",
            aspect="auto"
        )
        fig_heatmap.update_traces(
            text=np.round(matrix * 100, 1),
            texttemplate="%{text}%",
            hovertemplate="Marcador: %{y}-%{x}<br>Probabilidad: %{z}%<extra></extra>",
            textfont=dict(size=10)
        )
        fig_heatmap.update_layout(
            **PLOTLY_TRANSPARENT_LAYOUT,
            height=300,
            margin=dict(l=28, r=12, t=8, b=26),
            coloraxis_showscale=False,
            xaxis=dict(tickfont=dict(size=10)),
            yaxis=dict(tickfont=dict(size=10))
        )
        st.plotly_chart(fig_heatmap, use_container_width=True, config=PLOTLY_CONFIG)
        st.markdown('</div>', unsafe_allow_html=True)

    # 3. Save prediction widget
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h4>✍️ Ingresar Marcador Predicho</h4>', unsafe_allow_html=True)
    
    col_input_h, col_vs, col_input_a, col_save = st.columns([4, 1, 4, 3])
    
    with col_input_h:
        predicted_goals_h = st.number_input(
            f"Goles {home_team}:", 
            min_value=0, max_value=20, value=1, step=1, key="pred_h"
        )
    with col_vs:
        st.markdown("<div style='text-align:center; font-size:2rem; font-weight:700; margin-top:15px;'>-</div>", unsafe_allow_html=True)
    with col_input_a:
        predicted_goals_a = st.number_input(
            f"Goles {away_team}:", 
            min_value=0, max_value=20, value=1, step=1, key="pred_a"
        )
        
    with col_save:
        st.markdown("<div style='margin-top:25px;'></div>", unsafe_allow_html=True)
        if st.button("💾 Guardar Predicción"):
            # Prepare payload for saving locally and posting to n8n
            predicted_score_str = f"{predicted_goals_h}-{predicted_goals_a}"
            most_probable_score = f"{prediction['top_scores'][0][0][0]}-{prediction['top_scores'][0][0][1]}"
            most_probable_score_prob = prediction['top_scores'][0][1]
            
            payload = {
                "match_id": selected_match_id or "CUSTOM",
                "home_team": home_team,
                "away_team": away_team,
                "predicted_goals_home": predicted_goals_h,
                "predicted_goals_away": predicted_goals_a,
                "poisson_home_win_prob": round(float(prediction["home_win_prob"]), 4),
                "poisson_draw_prob": round(float(prediction["draw_prob"]), 4),
                "poisson_away_win_prob": round(float(prediction["away_win_prob"]), 4),
                "poisson_expected_home_goals": round(float(prediction["expected_home_goals"]), 3),
                "poisson_expected_away_goals": round(float(prediction["expected_away_goals"]), 3),
                "poisson_most_likely_score": most_probable_score,
                "poisson_most_likely_score_prob": round(float(most_probable_score_prob), 4)
            }
            
            # 1. Save prediction locally in JSON file
            local_success = save_prediction(payload)
            
            if local_success:
                st.balloons()
                st.success(f"✅ ¡Predicción guardada exitosamente en `{PREDICTIONS_FILE}`!")
            else:
                st.error("❌ Ocurrió un error al intentar guardar la predicción.")
                
    # Show Top 3 most likely scores in a banner
    st.markdown("<p style='font-weight:600; margin-bottom:5px;'>💡 Resultados más probables según el Modelo de Poisson:</p>", unsafe_allow_html=True)
    cols_top = st.columns(3)
    for idx, (score, prob) in enumerate(prediction["top_scores"][:3]):
        with cols_top[idx]:
            st.markdown(
                f"""
                <div class='score-pick-card'>
                    <span style='font-weight:700; color:#00E676;'>{score[0]} - {score[1]}</span><br>
                    <span style='font-size:0.8rem; color:#8892b0;'>{prob*100:.1f}% de probabilidad</span>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------
# TAB 2: MY PREDICTIONS DASHBOARD
# ------------------------------------------
with tab_dashboard:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h3>📊 Dashboard de Rendimiento</h3>', unsafe_allow_html=True)
    
    saved_preds = load_predictions()
    
    if not saved_preds:
        st.info("Aún no ha ingresado ninguna predicción. Utilice la pestaña 'Hub de Predicción' para registrar su primer pronóstico.")
    else:
        results_by_id, results_by_teams = _build_real_results_lookup(matches)
        schedule_by_id, schedule_by_teams = _build_match_schedule_lookup(matches)
        evaluated_preds = []
        for pred in saved_preds:
            real_result = _real_result_for_prediction(pred, results_by_id, results_by_teams)
            schedule_info = _schedule_for_prediction(pred, schedule_by_id, schedule_by_teams)
            evaluation = _evaluate_prediction(pred, real_result)
            evaluated_preds.append({
                **pred,
                "_real_result": real_result,
                "_schedule": schedule_info,
                "_evaluation": evaluation,
            })

        total_predictions = len(evaluated_preds)
        finished_evaluations = [p for p in evaluated_preds if p["_evaluation"]["category"] != "pending"]
        evaluated_count = len(finished_evaluations)
        pending_count = total_predictions - evaluated_count
        total_points = sum(p["_evaluation"]["points"] for p in evaluated_preds)
        avg_points = total_points / evaluated_count if evaluated_count else 0
        scored_count = sum(1 for p in finished_evaluations if p["_evaluation"]["points"] > 0)
        scored_pct = scored_count / evaluated_count * 100 if evaluated_count else 0

        kpi_points, kpi_evaluated, kpi_avg, kpi_scored, kpi_pending = st.columns(5)
        kpi_points.metric("Puntos totales", f"{total_points}")
        kpi_evaluated.metric("Predicciones evaluadas", f"{evaluated_count}")
        kpi_avg.metric("Puntos promedio", f"{avg_points:.2f}")
        kpi_scored.metric("% con puntos", f"{scored_pct:.0f}%")
        kpi_pending.metric("Pendientes", f"{pending_count}")

        st.markdown("<br>", unsafe_allow_html=True)

        category_config = {
            "exact": {"label": "Marcadores exactos", "short": "Exacto", "points": 5, "color": "#00E676"},
            "diff": {"label": "Diferencias acertadas", "short": "Diferencia", "points": 3, "color": "#00B0FF"},
            "winner": {"label": "Ganadores acertados", "short": "Ganador", "points": 2, "color": "#FFD54F"},
            "miss": {"label": "No acertadas", "short": "No acertada", "points": 0, "color": "#FF5252"},
            "pending": {"label": "Pendientes", "short": "Pendiente", "points": 0, "color": "#8892b0"},
        }

        cat_cols = st.columns(4)
        for col, category in zip(cat_cols, ["exact", "diff", "winner", "miss"]):
            category_preds = [p for p in evaluated_preds if p["_evaluation"]["category"] == category]
            count = len(category_preds)
            points = sum(p["_evaluation"]["points"] for p in category_preds)
            pct = count / evaluated_count * 100 if evaluated_count else 0
            config = category_config[category]
            col.markdown(
                f"""
                <div style='background:rgba(255, 255, 255, 0.025); border:1px solid rgba(255, 255, 255, 0.06); border-radius:12px; padding:14px; min-height:118px;'>
                    <div style='color:{config["color"]}; font-size:0.78rem; font-weight:700; text-transform:uppercase;'>{config["label"]}</div>
                    <div style='font-size:2rem; font-weight:800; color:#ffffff; line-height:1.15;'>{count}</div>
                    <div style='color:#8892b0; font-size:0.9rem;'>{points} pts · {pct:.0f}% evaluadas</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        chart_left, chart_right = st.columns([1, 1.45])
        category_rows = []
        for category in ["exact", "diff", "winner", "miss"]:
            count = sum(1 for p in evaluated_preds if p["_evaluation"]["category"] == category)
            if count:
                category_rows.append({
                    "Categoría": category_config[category]["short"],
                    "Predicciones": count,
                    "Color": category_config[category]["color"],
                })

        with chart_left:
            st.markdown("<h4>Distribución de aciertos</h4>", unsafe_allow_html=True)
            if category_rows:
                category_df = pd.DataFrame(category_rows)
                fig_category = px.pie(
                    category_df,
                    values="Predicciones",
                    names="Categoría",
                    hole=0.48,
                    color="Categoría",
                    color_discrete_map={row["Categoría"]: row["Color"] for row in category_rows},
                )
                fig_category.update_traces(textposition="inside", textinfo="percent+label")
                fig_category.update_layout(
                    **PLOTLY_TRANSPARENT_LAYOUT,
                    height=310,
                    margin=dict(l=10, r=10, t=10, b=10),
                    showlegend=False,
                )
                st.plotly_chart(fig_category, use_container_width=True, config=PLOTLY_CONFIG)
            else:
                st.info("Aún no hay predicciones evaluadas para graficar.")

        with chart_right:
            st.markdown("<h4>Evolución de puntos</h4>", unsafe_allow_html=True)
            timeline_rows = []
            for pred in evaluated_preds:
                timestamp = pred.get("timestamp")
                if pred["_evaluation"]["category"] == "pending" or not timestamp:
                    continue
                try:
                    registered_at = datetime.fromisoformat(timestamp)
                except ValueError:
                    continue
                timeline_rows.append({
                    "Registrado": registered_at,
                    "Partido": f"{pred.get('home_team')} vs {pred.get('away_team')}",
                    "Puntos": pred["_evaluation"]["points"],
                })

            if timeline_rows:
                timeline_df = pd.DataFrame(timeline_rows).sort_values("Registrado")
                timeline_df["Puntos acumulados"] = timeline_df["Puntos"].cumsum()
                fig_timeline = px.line(
                    timeline_df,
                    x="Registrado",
                    y="Puntos acumulados",
                    markers=True,
                    hover_data=["Partido", "Puntos"],
                )
                fig_timeline.update_traces(line=dict(color="#00E676", width=3), marker=dict(size=8, color="#00B0FF"))
                fig_timeline.update_layout(
                    **PLOTLY_TRANSPARENT_LAYOUT,
                    height=310,
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)", rangemode="tozero"),
                )
                st.plotly_chart(fig_timeline, use_container_width=True, config=PLOTLY_CONFIG)
            else:
                st.info("La evolución aparecerá cuando haya resultados reales para tus predicciones.")

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<h3>📋 Log de Predicciones</h3>', unsafe_allow_html=True)

        filter_options = {
            "Todas": None,
            "Marcador exacto": "exact",
            "Diferencia acertada": "diff",
            "Ganador acertado": "winner",
            "No acertadas": "miss",
            "Pendientes": "pending",
        }
        selected_filter = st.selectbox("Filtrar por categoría", list(filter_options.keys()), index=0)
        selected_category = filter_options[selected_filter]
        visible_preds = [
            p for p in evaluated_preds
            if selected_category is None or p["_evaluation"]["category"] == selected_category
        ]

        now = datetime.now()
        upcoming_preds = []
        played_preds = []
        for pred in visible_preds:
            scheduled_at = (pred.get("_schedule") or {}).get("scheduled_at")
            if pred["_real_result"] or (scheduled_at and scheduled_at <= now):
                played_preds.append(pred)
            else:
                upcoming_preds.append(pred)

        def _sort_datetime(pred, default):
            scheduled_at = (pred.get("_schedule") or {}).get("scheduled_at")
            if scheduled_at:
                return scheduled_at
            timestamp = pred.get("timestamp")
            if timestamp:
                try:
                    return datetime.fromisoformat(timestamp)
                except ValueError:
                    pass
            return default

        upcoming_preds = sorted(upcoming_preds, key=lambda p: _sort_datetime(p, datetime.max))
        played_preds = sorted(played_preds, key=lambda p: _sort_datetime(p, datetime.min), reverse=True)

        def render_prediction_log_card(pred, key_prefix):
            pred_id = pred.get("id")
            timestamp = pred.get("timestamp", "")
            if timestamp:
                try:
                    dt_obj = datetime.fromisoformat(timestamp)
                    formatted_time = dt_obj.strftime("%d/%m/%Y %H:%M")
                except ValueError:
                    formatted_time = timestamp
            else:
                formatted_time = "N/A"
                
            pred_score = f"{pred['predicted_goals_home']} - {pred['predicted_goals_away']}"
            poisson_score = pred.get("poisson_most_likely_score", "N/A")
            poisson_prob = pred.get("poisson_most_likely_score_prob", 0.0)
            real_result = pred["_real_result"]
            evaluation = pred["_evaluation"]
            real_score = (
                f"{real_result['home_score']} - {real_result['away_score']}"
                if real_result else "Pendiente"
            )
            result_source = real_result.get("source", "Resultado real") if real_result else "Pendiente"
            match_name = f"{pred['home_team']} vs {pred['away_team']}"
            scheduled_at = (pred.get("_schedule") or {}).get("scheduled_at")
            formatted_match_time = _format_match_datetime(scheduled_at)

            col_info, col_delete = st.columns([10, 2])
            
            with col_info:
                st.markdown(
                    f"""
                    <div style='background:rgba(255, 255, 255, 0.02); border:1px solid rgba(255, 255, 255, 0.05); padding:15px; border-radius:12px; margin-bottom:10px;'>
                        <div style='display:flex; justify-content:space-between;'>
                            <span style='font-weight:700; font-size:1.1rem; color:#ffffff;'>⚽ {escape(str(match_name))}</span>
                            <span style='font-size:0.75rem; color:#8892b0;'>Partido: {formatted_match_time}</span>
                        </div>
                        <div style='margin-top:10px; display:flex; gap:20px; flex-wrap:wrap;'>
                            <div>🗣️ <b>Tu Predicción:</b> <span style='font-size:1.2rem; color:#00E676;'>{pred_score}</span></div>
                            <div>🏁 <b>Resultado real:</b> <span style='font-size:1.2rem; color:#ffffff;'>{real_score}</span><span style='font-size:0.78rem; color:#8892b0;'> ({escape(str(result_source))})</span></div>
                            <div><span style='display:inline-block; padding:4px 10px; border-radius:20px; font-size:0.75rem; font-weight:700; text-transform:uppercase; color:{evaluation["color"]}; background:{evaluation["background"]}; border:1px solid {evaluation["border"]};'>{evaluation["label"]} · {evaluation["points"]} pts</span></div>
                            <div>🤖 <b>Model Poisson:</b> <span style='color:#00B0FF;'>{poisson_score}</span> ({poisson_prob*100:.1f}%)</div>
                            <div>🕒 <b>Registrado:</b> {formatted_time}</div>
                            <div>📈 <b>Probabilidades Poisson:</b> {escape(str(pred['home_team']))} {pred['poisson_home_win_prob']*100:.0f}% | Empate {pred['poisson_draw_prob']*100:.0f}% | {escape(str(pred['away_team']))} {pred['poisson_away_win_prob']*100:.0f}%</div>
                        </div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            
            with col_delete:
                st.markdown("<div style='margin-top:15px;'></div>", unsafe_allow_html=True)
                if st.button("❌ Eliminar", key=f"del_{key_prefix}_{pred_id}", help="Eliminar predicción del archivo"):
                    if delete_prediction(pred_id):
                        st.success("Predicción eliminada.")
                        st.rerun()

        upcoming_col, played_col = st.columns(2)
        with upcoming_col:
            st.markdown(f"<h4>⏭️ Predicciones por venir ({len(upcoming_preds)})</h4>", unsafe_allow_html=True)
            st.caption("Ordenadas por el partido más cercano primero.")
            if upcoming_preds:
                for pred in upcoming_preds:
                    render_prediction_log_card(pred, "upcoming")
            else:
                st.info("No hay predicciones futuras para este filtro.")

        with played_col:
            st.markdown(f"<h4>✅ Ya jugados ({len(played_preds)})</h4>", unsafe_allow_html=True)
            st.caption("Ordenadas desde el partido más reciente al más antiguo.")
            if played_preds:
                for pred in played_preds:
                    render_prediction_log_card(pred, "played")
            else:
                st.info("No hay partidos jugados para este filtro.")
                        
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------
# TAB 3: TEAM RATINGS & STATS
# ------------------------------------------
with tab_ratings:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h3>📊 Ratings de Fuerza Computados</h3>', unsafe_allow_html=True)
    st.markdown(
        """
        El modelo Poisson utiliza dos valores críticos para cada equipo:
        - **Fuerza de Ataque**: Capacidad goleadora en relación con el promedio (valores superiores a 1.0 son mejores).
        - **Fuerza de Defensa**: Nivel defensivo en relación con el promedio (valores inferiores a 1.0 representan mejores defensas).
        """
    )
    
    # Render ratings dataframe
    ratings_records = []
    for team, rating in team_ratings.items():
        ratings_records.append({
            "Equipo": team,
            "Fuerza Ataque": rating.get("attack", 1.0),
            "Fuerza Defensa": rating.get("defense", 1.0),
            "Partidos Jugados (API)": rating.get("games_played", 0)
        })
        
    df_ratings = pd.DataFrame(ratings_records).sort_values("Fuerza Ataque", ascending=False)
    
    # 1. Plot comparison using Plotly
    fig_comparison = px.scatter(
        df_ratings,
        x="Fuerza Defensa",
        y="Fuerza Ataque",
        text="Equipo",
        title="Fuerzas de Ataque vs Defensa (Mundial 2026)",
        labels={"Fuerza Defensa": "Fuerza Defensa (Menos es mejor)", "Fuerza Ataque": "Fuerza Ataque (Más es mejor)"},
        color="Partidos Jugados (API)",
        color_continuous_scale="Viridis",
        height=500
    )
    fig_comparison.update_traces(textposition='top center', marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
    # Add quadrants lines
    fig_comparison.add_vline(x=1.0, line_dash="dash", line_color="gray", opacity=0.5)
    fig_comparison.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Customise chart layouts
    fig_comparison.update_layout(
        **PLOTLY_TRANSPARENT_LAYOUT,
        margin=dict(l=10, r=10, t=50, b=20),
        height=430,
        legend=dict(orientation="h"),
        coloraxis_colorbar=dict(thickness=10)
    )
    st.plotly_chart(fig_comparison, use_container_width=True, config=PLOTLY_CONFIG)
    
    # Display table in columns
    st.markdown("<h4>Tabla de Fuerza de los Equipos</h4>", unsafe_allow_html=True)
    st.dataframe(
        df_ratings.style.background_gradient(cmap="RdYlGn_r", subset=["Fuerza Defensa"])
                        .background_gradient(cmap="RdYlGn", subset=["Fuerza Ataque"]),
        use_container_width=True,
        hide_index=True,
        height=420
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------
# TAB 4: TOURNAMENT FIXTURES
# ------------------------------------------
with tab_fixtures:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h3>📅 Fixtures y Resultados oficiales (football-data.org)</h3>', unsafe_allow_html=True)
    
    # Show filters
    filter_status = st.multiselect(
        "Filtrar por Estado de Partido:",
        options=["FINISHED", "SCHEDULED", "TIMED", "LIVE", "IN_PLAY"],
        default=["FINISHED", "SCHEDULED"]
    )
    
    filtered_matches = [m for m in matches if m.get("status") in filter_status]
    
    if not filtered_matches:
        st.info("No se encontraron partidos para el filtro seleccionado.")
    else:
        table_data = []
        for m in filtered_matches:
            date_str = m.get("utcDate", "")
            if date_str:
                dt = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
                formatted_date = dt.strftime("%d/%m/%Y %H:%M")
            else:
                formatted_date = "Por definir"
                
            home = m["homeTeam"]["name"]
            away = m["awayTeam"]["name"]
            status = m.get("status", "")
            
            score = m.get("score", {})
            full_time = score.get("fullTime", {})
            home_goals = full_time.get("home")
            away_goals = full_time.get("away")
            
            goals_display = f"{home_goals} - {away_goals}" if home_goals is not None else "vs"

            table_data.append({
                "Fecha": formatted_date,
                "Local": home,
                "Marcador": goals_display,
                "Visitante": away,
                "Estado": status
            })
            
        df_table = pd.DataFrame(table_data)
        st.dataframe(df_table, use_container_width=True, hide_index=True, height=520)

    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------
# TAB 5: SIMULACIONES MONTE CARLO
# ------------------------------------------
with tab_sim:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h3>🎲 Simulación Monte Carlo — FIFA World Cup 2026</h3>', unsafe_allow_html=True)
    st.markdown(
        "Corre miles de simulaciones completas del torneo (fase de grupos + eliminatorias) "
        "para estimar la probabilidad de cada selección de ganar, llegar a la final, etc.",
        unsafe_allow_html=False
    )

    # --- Controls ---
    col_ctrl1, col_ctrl2 = st.columns([1, 2])
    with col_ctrl1:
        n_sims = st.select_slider(
            "Número de simulaciones",
            options=[1_000, 5_000, 10_000, 25_000, 50_000],
            value=10_000,
            help="Más simulaciones = mayor precisión, pero tarda más."
        )
    with col_ctrl2:
        st.info(
            f"**Formato WC 2026:** 12 grupos de 4 equipos · "
            f"Top 2 de cada grupo + 8 mejores 3eros = 32 equipos en fase eliminatoria · "
            f"R32 → R16 → QF → SF → Final"
        )

    run_btn = st.button("▶ Correr Simulación", type="primary", use_container_width=False)

    if run_btn:
        with st.spinner(f"Corriendo {n_sims:,} simulaciones completas del Mundial 2026..."):
            sim_results = run_monte_carlo(
                team_ratings={},  # predictor uses ML models internally
                n_simulations=n_sims,
                groups=WC2026_GROUPS
            )

        # Build DataFrame
        rows = []
        for team, probs in sim_results.items():
            rows.append({
                "Equipo": team,
                "Campeón %":    round(probs["winner"] * 100, 2),
                "Final %":      round(probs["final"]  * 100, 2),
                "Semifinal %":  round(probs["sf"]     * 100, 2),
                "Cuartos %":    round(probs["qf"]     * 100, 2),
                "Octavos %":    round(probs["r16"]    * 100, 2),
                "Clasifica %":  round(probs["r32"]    * 100, 2),
            })
        df_sim = pd.DataFrame(rows).sort_values("Campeón %", ascending=False).reset_index(drop=True)
        df_sim.index += 1  # 1-based ranking

        # --- CHART: Champion probability top 20 ---
        top20 = df_sim.head(20)
        fig_champ = go.Figure(go.Bar(
            x=top20["Equipo"],
            y=top20["Campeón %"],
            marker_color=[
                "#FFD700" if i == 0 else "#C0C0C0" if i == 1 else "#CD7F32" if i == 2
                else "#4A90D9"
                for i in range(len(top20))
            ],
            text=[f"{v:.1f}%" for v in top20["Campeón %"]],
            textposition="outside",
        ))
        fig_champ.update_layout(
            **PLOTLY_TRANSPARENT_LAYOUT,
            title=f"Probabilidad de ganar el Mundial 2026 (n={n_sims:,})",
            xaxis_title="",
            yaxis_title="Probabilidad (%)",
            yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
            xaxis=dict(tickangle=-35, tickfont=dict(size=10)),
            margin=dict(l=10, r=10, t=52, b=80),
            height=390,
        )
        st.plotly_chart(fig_champ, use_container_width=True, config=PLOTLY_CONFIG)

        # --- CHART: Stacked bar for round progression top 16 ---
        top16 = df_sim.head(16)
        fig_rounds = go.Figure()
        round_cols = ["Clasifica %", "Octavos %", "Cuartos %", "Semifinal %", "Final %", "Campeón %"]
        colors = ["#2d5a8e", "#3a7bd5", "#5b9bd5", "#f0a500", "#e05c28", "#FFD700"]
        labels = ["Clasifica", "Octavos", "Cuartos", "Semifinal", "Final", "Campeón"]

        prev = np.zeros(len(top16))
        for col, color, label in zip(round_cols, colors, labels):
            vals = top16[col].values
            fig_rounds.add_trace(go.Bar(
                name=label,
                x=top16["Equipo"],
                y=vals - prev,
                marker_color=color,
                text=[f"{v:.0f}%" if (v - p) > 3 else "" for v, p in zip(vals, prev)],
                textposition="inside",
            ))
            prev = vals.copy()

        fig_rounds.update_layout(
            **PLOTLY_TRANSPARENT_LAYOUT,
            barmode="stack",
            title="Progresión por ronda — Top 16 equipos",
            xaxis_title="",
            yaxis_title="Probabilidad acumulada (%)",
            xaxis=dict(tickangle=-35, tickfont=dict(size=10)),
            yaxis=dict(gridcolor="rgba(255,255,255,0.1)", range=[0, 105]),
            legend=dict(orientation="h", y=1.14, font=dict(size=10)),
            margin=dict(l=10, r=10, t=68, b=80),
            height=400,
        )
        st.plotly_chart(fig_rounds, use_container_width=True, config=PLOTLY_CONFIG)

        # --- TABLE: Full results ---
        st.markdown("#### Tabla completa de probabilidades")
        st.dataframe(
            df_sim.style.background_gradient(subset=["Campeón %"], cmap="YlOrRd"),
            use_container_width=True,
            height=500,
        )

        # --- Groups display ---
        st.markdown("#### Grupos WC 2026 utilizados en la simulación")
        st.caption("Grupos cargados desde data-worldcup/wc2026_schedule.json (sorteo oficial FIFA).")
        gcols = st.columns(4)
        group_list = list(WC2026_GROUPS.items())
        for i, (grp, teams) in enumerate(group_list):
            with gcols[i % 4]:
                st.markdown(f"**Grupo {grp}**")
                for t in teams:
                    champ_pct = sim_results.get(t, {}).get("winner", 0) * 100
                    st.markdown(f"- {t} `{champ_pct:.1f}%`")

    else:
        st.markdown(
            "Presiona **▶ Correr Simulación** para generar las probabilidades. "
            "10,000 simulaciones tarda ~30 segundos."
        )

    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------
# TAB 6: ACTUALIZAR DATOS
# ------------------------------------------
with tab_data:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h3>🔄 Actualizar Datos del Modelo</h3>', unsafe_allow_html=True)
    st.markdown(
        "Incorpora resultados nuevos (amistosos y partidos del Mundial) al corpus "
        "de entrenamiento. Los datos más recientes pesan más en el modelo.",
        unsafe_allow_html=False
    )

    # --- Estado del corpus ---
    status = data_updater.corpus_status()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Partidos en el corpus", f"{status['total_matches']:,}")
    c2.metric("Último partido", status["last_match_date"])
    c3.metric("Añadidos manualmente", status["extra_matches"])
    c4.metric("Último entrenamiento", status["last_train"])

    st.markdown('</div>', unsafe_allow_html=True)

    # --- Sincronizar Mundial vía API ---
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h4>📡 Sincronizar Mundial (API football-data.org)</h4>', unsafe_allow_html=True)
    st.markdown(
        "Descarga automáticamente los partidos **finalizados** del Mundial 2026 "
        "y los agrega al corpus (sin duplicar).",
        unsafe_allow_html=False
    )
    if st.button("📥 Sincronizar resultados del Mundial", use_container_width=False):
        if not api_key:
            st.warning("Configura tu API key de football-data.org en la barra lateral primero.")
        else:
            with st.spinner("Consultando la API..."):
                ok, msg, n_added = data_updater.fetch_api_results(api_key)
            if ok:
                st.success(msg)
                if n_added > 0:
                    st.info("Recuerda **Reentrenar modelos** abajo para usar los datos nuevos.")
            else:
                st.error(msg)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Añadir partido manual ---
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h4>✍️ Añadir partido manual</h4>', unsafe_allow_html=True)
    st.markdown(
        "Para amistosos u otros resultados que la API no incluya.",
        unsafe_allow_html=False
    )

    team_options = sorted(set(playable_teams) | set(ALL_TEAMS))

    with st.form("add_match_form", clear_on_submit=True):
        fc1, fc2 = st.columns(2)
        with fc1:
            m_date = st.date_input("Fecha del partido")
            m_home = st.selectbox("Equipo local", team_options, index=0)
            m_home_score = st.number_input("Goles local", min_value=0, max_value=30, value=0, step=1)
            m_tournament = st.text_input("Torneo", value="Friendly")
        with fc2:
            m_country = st.text_input("País sede (opcional)", value="")
            m_away = st.selectbox("Equipo visitante", team_options, index=1)
            m_away_score = st.number_input("Goles visitante", min_value=0, max_value=30, value=0, step=1)

        submitted = st.form_submit_button("➕ Añadir partido")
        if submitted:
            ok, msg = data_updater.add_match(
                m_date, m_home, m_away, m_home_score, m_away_score,
                m_tournament, m_country
            )
            if ok:
                st.success(msg)
                st.info("Recuerda **Reentrenar modelos** abajo para usar los datos nuevos.")
            else:
                st.error(msg)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Reentrenar modelos ---
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h4>🤖 Reentrenar modelos</h4>', unsafe_allow_html=True)
    st.markdown(
        "Vuelve a entrenar los modelos con todos los datos (incluidos los nuevos). "
        "Tarda ~30 segundos.",
        unsafe_allow_html=False
    )
    if st.button("🔁 Reentrenar ahora", type="primary", use_container_width=False):
        with st.spinner("Reentrenando modelos con los datos actualizados..."):
            ok, msg = data_updater.retrain()
        if ok:
            st.cache_data.clear()
            st.success(msg + " Las predicciones y simulaciones ya usan los datos nuevos.")
        else:
            st.error(msg)
    st.markdown('</div>', unsafe_allow_html=True)
