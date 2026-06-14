# Contexto del repositorio — Polla Mundial 2026

App Streamlit para pronosticar partidos del Mundial 2026. Combina **Random Forest** (goles esperados) + **Poisson** (probabilidades de resultado y matriz de marcadores). Incluye polla personal con scoring, simulación Monte Carlo del torneo y reentrenamiento desde la UI.

## Comandos esenciales

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py                    # desarrollo local
python train_model.py                     # reentrenar RF → models/
python backtest.py                        # walk-forward backtest
python fetch_squad_values.py              # opcional: regenerar squad_values.csv
```

Deploy: Railway vía `Procfile`. Ver `RAILWAY_DEPLOY.md`.

## Arquitectura (flujo de datos)

```
Fuentes de partidos                    Corpus de entrenamiento
─────────────────────                  ───────────────────────
football-data.org API ──┐                data-worldcup/data-csv/matches.csv
wc2026_schedule.json  ──┼→ utils.py      International Football Results/all_matches.csv
extra_matches.csv     ──┘                data-worldcup/extra_matches.csv
                                                    ↓
                                            data_prep.py (features leak-free)
                                                    ↓
                                            train_model.py → models/*.pkl, *.json
                                                    ↓
app.py ← predictor.py (RF xG + Poisson) ← simulation.py (Monte Carlo)
         utils.py (predicciones JSON)
```

## Módulos — responsabilidad única

| Archivo | Rol |
|---------|-----|
| `app.py` | UI Streamlit (6 tabs), CSS glassmorphism, PWA, scoring de polla |
| `predictor.py` | `predict_match()`: RF para λ, Poisson para W/D/L y matriz; fallback Poisson puro |
| `simulation.py` | Monte Carlo WC 2026: grupos + bracket desde `wc2026_schedule.json` |
| `data_prep.py` | Corpus combinado, decay temporal, features as-of, confederaciones |
| `train_model.py` | Entrena 2× RandomForestRegressor, exporta a `models/` |
| `data_updater.py` | Sin Streamlit: añadir partidos, sync API, retrain, `corpus_status()` |
| `utils.py` | API football-data.org + fallback JSON, CRUD `predictions.json` |
| `team_names.py` | `normalize_team_name()`, `is_placeholder_team()` — **usar siempre** |
| `elo.py` | Elo secuencial leak-free para features |
| `dixon_coles.py` | Corrección Dixon-Coles; **rho=0 en producción** (independiente Poisson) |
| `backtest.py` | Evaluación walk-forward; no importado por la app |

## Pipeline de predicción (`predict_match`)

1. Si existen `models/home_model.pkl`, `away_model.pkl`, `feature_columns.json`, `team_stats.json` → **modo ML**.
2. RF predice goles esperados (λ_home, λ_away) con features de `build_ml_feature_values()`.
3. Matriz de marcadores vía `dc_score_matrix(λ_h, λ_a, rho=0)`.
4. Sin modelos → fallback Poisson clásico: `attack × defense × tournament_avg` con `SEED_RATINGS`.
5. Los sliders de ataque/defensa en la UI **solo afectan el fallback Poisson**, no el modo ML.

Features ML (21 cols, ver `models/feature_columns.json`): promedios goles, forma reciente (5 partidos), Elo, flags anfitrión (USA/Mexico/Canada), `is_world_cup`, valor plantilla, fuerza confederación.

## Datos clave

- **Calendario WC 2026**: `data-worldcup/wc2026_schedule.json` — grupos, bracket knockout, 104 partidos.
- **Corpus histórico**: mundiales (`matches.csv`) + internacionales (`all_matches.csv`, ≥1990 para entrenar RF).
- **Partidos nuevos**: `data-worldcup/extra_matches.csv` (manual o sync API); deduplicados por `(fecha, home, away)` normalizado.
- **Modelos entrenados**: `models/` — `.pkl` + JSON de lookup. Commiteados al repo.
- **Squad values**: `data-worldcup/squad_values.csv` → `models/squad_values_2026.json` al entrenar.
- **Paquete R `data-worldcup/`**: dataset histórico FIFA; la app Python solo usa CSV/JSON, no ejecuta R.

## Variables de entorno

| Variable | Uso |
|----------|-----|
| `FOOTBALL_DATA_API_KEY` | API football-data.org (sidebar o `.env`) |
| `PREDICTIONS_FILE` | Ruta JSON de predicciones; default `/app/storage/predictions.json` si existe Volume |

Sin API key → calendario offline desde `wc2026_schedule.json` (sin resultados en vivo).

## UI (`app.py`) — tabs

1. **Predicciones**: dashboard polla, scoring (5 pts exacto, 3 diff, 2 ganador).
2. **Pronóstico**: elegir partido, ver probabilidades y guardar predicción.
3. **Ratings**: fuerzas Poisson calculadas de partidos FINISHED.
4. **Calendario**: fixtures API/offline.
5. **Simular**: Monte Carlo (1k–50k sims, ~30s con 10k).
6. **Datos**: sync API, partido manual, reentrenar modelos.

## Convenciones al modificar

- **Nombres de equipos**: siempre `normalize_team_name()` antes de comparar o predecir. Aliases en `team_names.py` y `elo.INTL_TEAM_NAME_MAP`.
- **Features leak-free**: en entrenamiento usar `build_asof_features()`; nunca mezclar stats globales como features de entrenamiento.
- **Cambiar features**: actualizar `train_model.py` BASE_FEATURE_COLS, `predictor.build_ml_feature_values()`, reentrenar, verificar que `feature_columns.json` coincide con `model.n_features_in_`.
- **Lógica sin UI**: poner en `data_updater.py` o módulos puros; `app.py` solo orquesta.
- **Cache API**: `fetch_wc_matches_api` tiene `@st.cache_data(ttl=600)` y rate limit 9 req/min.
- **Idioma UI**: español. Código/comentarios: inglés (convención existente).
- **No commitear**: `.env`, `.venv/`, datasets crudos pesados (ver `.gitignore`, `.railwayignore`).

## Scoring de polla (referencia)

| Acierto | Puntos |
|---------|--------|
| Marcador exacto | 5 |
| Misma diferencia de goles | 3 |
| Mismo ganador/empate | 2 |
| Fallo | 0 |

Evaluación en `_evaluate_prediction()` contra resultados API + `extra_matches.csv`.

## Simulación WC 2026

- 12 grupos × 4 equipos → top 2 + 8 mejores terceros → 32 en eliminatorias.
- Grupos/bracket parseados al importar `simulation.py` desde `wc2026_schedule.json`.
- Lambdas precomputados en batch para todas las parejas (evita I/O en el loop).
- Draws permitidos en fase de grupos; eliminatorias resuelven empates (penales simulados).

## Pitfalls conocidos

- **Dixon-Coles desactivado** (`rho=0`): backtest mostró peor rendimiento en mundiales.
- **Placeholders de bracket** (`1A`, `W73`, `3A/B/C/D/F`): filtrar con `is_placeholder_team()`.
- **Mismatch de features**: si cambias columnas sin reentrenar, `predict_match` cae a Poisson fallback.
- **PWA**: service worker cachea assets; la app Streamlit **no** funciona offline completo.
- **`International Football Results/`**: carpeta con espacio en el path; respetar al referenciar.
- **Reentrenar desde UI** llama `data_updater.retrain()` → `train_model.train_models()` y limpia caché Streamlit.

## Archivos que no tocar sin motivo

- `data-worldcup/data-csv/*.csv` — dataset histórico estático.
- `static/` — PWA (manifest, sw.js, offline.html).
- `predictions.json` — seed local; en producción vive en Volume.
