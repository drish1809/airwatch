# 🌬️ AirWatch — Real-Time Air Quality Monitoring & Prediction

> **Production-ready end-to-end Data Science project** demonstrating data engineering, ML pipelines, and interactive dashboards.

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Quick Start (Demo Mode)](#quick-start-demo-mode)
- [Live Data Collection Setup](#live-data-collection-setup)
- [Project Structure](#project-structure)
- [Feature Details](#feature-details)
- [Deployment](#deployment)
- [Recruiter Notes](#recruiter-notes)

---

## 🎯 Project Overview

AirWatch is a **full data product** that covers the entire DS/DE lifecycle:

| Layer | Technology | What it does |
|---|---|---|
| **Data Engineering** | OpenWeatherMap API + APScheduler + SQLite | Automated collection every 30 min |
| **Data Processing** | pandas + scikit-learn | Cleaning, lag features, rolling stats |
| **Machine Learning** | 6 models + RandomizedSearchCV | Auto model selection & tuning |
| **Dashboard** | Streamlit + Plotly | 6 interactive pages |
| **Deployment** | Streamlit Cloud / Render / HuggingFace | One-command deploy |

---

## 🏗️ Architecture

```
Live API ──► Scheduler ──► SQLite DB ──► Processor ──► ML Trainer
   (every 30 min)              │                            │
                               └──────────────┐        best_model.pkl
                                              ▼
                               Streamlit Dashboard (app/main.py)
                                    ├── 📊 City Dashboard
                                    ├── 🌍 City Comparison
                                    ├── 🔮 AQI Prediction
                                    ├── ⚠️ Health & Safety
                                    ├── 🗺️ Map View
                                    └── 🔬 Anomaly Detection
```

---

## ⚡ Quick Start (Demo Mode — No API Key Needed)

```bash
# 1. Clone / download the project
cd air_quality_project

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate synthetic data + train model
python run.py --demo

# 5. Launch dashboard
python run.py --app
```

Open **http://localhost:8501** in your browser. 🎉

---

## 📡 Live Data Collection Setup

### Step 1 — Get a free API key
Sign up at [openweathermap.org](https://openweathermap.org/api) → **Air Pollution API** (free tier: 60 calls/min).

### Step 2 — Add your key
Edit `config/config.yaml`:
```yaml
api:
  openweather_api_key: "your_actual_key_here"
```

### Step 3 — Start collecting
```bash
# Collect data every 30 minutes (runs indefinitely)
python run.py --collect

# Or run everything: collect + train + dashboard
python run.py --all
```

---

## 📁 Project Structure

```
air_quality_project/
│
├── config/
│   └── config.yaml             # All settings (API key, cities, scheduler)
│
├── src/
│   ├── __init__.py
│   ├── utils.py                # Logging, config loading, AQI helpers
│   ├── collector.py            # API data collection + APScheduler
│   ├── processor.py            # Cleaning + feature engineering pipeline
│   ├── trainer.py              # 6-model training + hyperparameter tuning
│   ├── predictor.py            # Inference wrapper for saved model
│   ├── anomaly.py              # Isolation Forest + spike alerts
│   └── demo_data.py            # Synthetic 30-day data generator
│
├── app/
│   └── main.py                 # Streamlit dashboard (6 pages)
│
├── data/
│   ├── raw/                    # CSV exports
│   ├── processed/              # Feature-engineered data
│   └── air_quality.db          # SQLite database
│
├── models/
│   ├── best_model.pkl          # Serialized best model
│   ├── scaler.pkl              # Feature scaler
│   └── metadata.json           # Model info + metrics + feature importance
│
├── logs/
│   └── app.log                 # Rotating log file
│
├── .streamlit/
│   └── config.toml             # Dark theme + server settings
│
├── run.py                      # Unified CLI entry point
├── requirements.txt
└── README.md
```

---

## 🔬 Feature Details

### Data Collection (`src/collector.py`)
- **12 global cities** out of the box (configurable via YAML)
- **Retry logic** with exponential-like back-off (3 attempts per city)
- **APScheduler** runs the collection job as a daemon thread
- **SQLite** with `UNIQUE(timestamp, city)` to prevent duplicate rows
- **CSV export** after every run for easy analysis

### Feature Engineering (`src/processor.py`)
| Feature type | Examples |
|---|---|
| Calendar | hour, day_of_week, month, is_weekend, is_rush_hour |
| Pollutant ratios | pm_ratio (PM2.5/PM10), no_no2_ratio |
| Lag features | pm2_5_lag_1/2/3, aqi_lag_1/2/3 |
| Rolling stats | pm2_5_rolling_mean_3, aqi_rolling_std_3 |
| Encoded | city_encoded (LabelEncoder) |

### ML Pipeline (`src/trainer.py`)
| Model | Library |
|---|---|
| Linear Regression | sklearn |
| Ridge / Lasso | sklearn |
| Random Forest | sklearn |
| Gradient Boosting | sklearn |
| Extra Trees | sklearn |

- **Auto-selection** by RMSE on hold-out test set
- **RandomizedSearchCV** hyperparameter tuning for top models
- **Cross-validation** scores reported alongside test metrics
- **Artefacts saved**: `best_model.pkl`, `scaler.pkl`, `metadata.json`

### Anomaly Detection (`src/anomaly.py`)
- **Isolation Forest** with configurable contamination rate
- **Spike alerts** — city AQI delta ≥ 50 within one interval
- Severity levels: MEDIUM / HIGH / CRITICAL

---

## 🚀 Deployment

### Option A — Streamlit Cloud (Recommended — free)
1. Push project to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set `app/main.py` as entrypoint
4. Add `OPENWEATHER_API_KEY` in Secrets (optional — demo mode works without)
5. Click **Deploy**

### Option B — Render
```yaml
# render.yaml
services:
  - type: web
    name: airwatch
    env: python
    buildCommand: "pip install -r requirements.txt && python run.py --demo"
    startCommand: "streamlit run app/main.py --server.port $PORT --server.headless true"
```

### Option C — HuggingFace Spaces
1. Create a new Space with **Streamlit** SDK
2. Upload all project files
3. Set `app_file = app/main.py` in README YAML header

### Option D — Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
RUN python run.py --demo
EXPOSE 8501
CMD ["streamlit", "run", "app/main.py", "--server.headless", "true"]
```

```bash
docker build -t airwatch .
docker run -p 8501:8501 airwatch
```

---

## 💼 Recruiter Notes

This project demonstrates:

| Skill | Evidence |
|---|---|
| **Data Engineering** | Automated REST API ingestion, SQLite schema design, retry/error handling, scheduler |
| **Data Processing** | Multi-step cleaning pipeline, lag/rolling feature engineering, imputation strategies |
| **Machine Learning** | 6-model comparison, hyperparameter tuning, cross-validation, artefact management |
| **Software Engineering** | Modular package structure, logging, config-driven design, CLI, docstrings |
| **Visualisation** | Plotly charts (bar, line, scatter, heatmap, radar, gauge, map), Streamlit layout |
| **Deployment** | Multi-platform deploy guide, Docker, cloud-ready configuration |
| **Anomaly Detection** | Isolation Forest + domain-specific spike alerting |

---

## 📄 License
MIT — free to use, modify, and distribute.
