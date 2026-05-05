
# AI-Driven Air Quality Monitoring and Forecasting for Almaty

This project is a comprehensive web-based system designed to monitor and predict air quality in Almaty. By combining real-time pollution metrics with meteorological data, the platform reports the current PM2.5 situation and generates two distinct types of forecasts:

- A short-term spatial forecast across the city using a Spatial-Temporal Graph Convolutional Network (STGCN).
- A daily average city-wide PM2.5 forecast using CatBoost.

The primary objective is not just to report current air quality, but to proactively estimate where smog will migrate and what the overall pollution baseline will be tomorrow.

## Key Features

- Real-time display of current PM2.5 and PM10 levels.
- Next-day PM2.5 forecasting for the entire city.
- Spatial modeling of pollution distribution across 24 anchor stations in Almaty.
- Real-time data imputation to seamlessly recover missing values if IoT sensors go offline.
- Interactive web interface built with FastAPI and Jinja2.
- Support for dedicated pages: forecast, historical data, scenario simulation, and an interactive map.

## System Overview

At a high level, the system operates through a straightforward data pipeline:

1. The application fetches fresh data from external APIs.
2. If a specific sensor is unresponsive, the system gracefully recovers its value using data from neighboring stations.
3. The processed data is fed into two distinct models:
   - One calculates the city's average pollution level for the following day.
   - The other calculates how the pollution will distribute across specific districts in the upcoming hours.
4. The backend aggregates these predictions and renders a comprehensive, user-friendly web page featuring metrics, actionable recommendations, and historical context.

## Architecture

The project follows a modular architecture, separating the FastAPI web layer from the service layer and model artifacts.

### 1. Web Layer
**Entry point:** `app/main.py`

This module is responsible for:
- FastAPI initialization.
- Pre-loading machine learning models into memory at startup.
- Managing routing (`/`, `/forecast`, `/history`, `/simulation`, `/map`).
- Passing context variables to Jinja2 HTML templates.

The web server acts purely as a coordinator; it does not train models or house heavy data-processing logic.

### 2. Services
Located in `app/services/`:

- `fetcher.py`: Retrieves data from Open-Meteo and OpenAQ.
- `imputer.py`: Recovers missing values for the 24 anchor stations using nearest-neighbor logic (Haversine distance).
- `catboost_service.py`: Engineers features and executes the daily forecast.
- `stgcn_service.py`: Loads the graph neural network and executes the spatial forecast.
- `data_manager.py`: Manages and updates local CSV datasets.
- `history_service.py`: Prepares historical data for frontend visualization.

### 3. Data Storage
Located in the `data/` directory.
Contains historical tables for training, time-series matrices by station, and station coordinate mapping.

### 4. Models
Located in the `models/` directory.
Contains pre-trained artifacts:
- `catboost_final.cbm`: The trained CatBoost model.
- `stgcn_almaty.pth`: The STGCN weights.
- `adj_matrix.npy`: The adjacency matrix defining spatial relationships between stations.
- `pm25_scaler.pkl`: The scaler used for data normalization.

## Machine Learning Models

### CatBoost (City-wide Daily Forecast)
Predicts the average PM2.5 level across Almaty for the next day.
**Inputs:**
- Current and lagged PM2.5/PM10 values.
- Meteorological features.
- Derived physical features (e.g., air stagnation flag, ventilation index, thermal inversion potential).
- Calendar and temporal features.

### STGCN (Spatial Forecast)
Predicts spatial distribution across 24 anchor stations.
**Inputs & Logic:**
- Each station acts as a node in a graph.
- Relationships (edges) between stations are defined by an adjacency matrix based on geographical distance.
- The model captures both the physical neighborhood effects and temporal dynamics to output a localized PM2.5 forecast for each node.

## Spatial Imputer Mechanism

IoT air quality sensors are notoriously unstable; devices frequently disconnect or return empty payloads. If the system required all 24 anchor stations to be online simultaneously, the forecasting pipeline would frequently break.

To solve this, the `SpatialImputer`:
- Identifies nearby active stations when an anchor point drops offline.
- Incrementally expands its search radius.
- Calculates a distance-weighted average based on the closest available "donor" sensors.
- Falls back to the city-wide background average if no local donors are found.

This ensures the graph neural network always receives a complete tensor, making the system highly resilient to hardware failures.

## Request Lifecycle

When a user navigates to the `/forecast` endpoint:
1. FastAPI receives the HTTP request.
2. `fetcher.py` pulls the latest weather and air quality data.
3. `data_manager.py` updates local records if necessary.
4. `CatBoostRunner` computes the next-day average.
5. `STGCNRunner` prepares the graph input and computes the spatial forecast.
6. Results are aggregated, packed into an HTML template, and served to the client.

## Project Structure

```text
app/
  main.py
  services/
    catboost_service.py
    stgcn_service.py
    fetcher.py
    imputer.py
    data_manager.py
    history_service.py
  templates/
  static/

data/
models/
requirements.txt
Dockerfile
docker-compose.yml
README.md
```

## Dependencies

- **Web & API:** `fastapi`, `uvicorn`, `requests`, `httpx`
- **Templates & Validation:** `jinja2`, `pydantic`
- **Data Processing:** `numpy`, `pandas`, `scikit-learn`
- **Machine Learning:** `catboost`, `torch`
- **Mapping:** `folium`

## Installation and Setup

**Recommended Python version:** `3.10` or `3.11`.
*(Note: Newer Python versions may lack stable pre-compiled wheels for PyTorch and CatBoost).*

### Local Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
cd app
python main.py
```

The application will be accessible at `http://127.0.0.1:8000`.

### Docker Setup (Recommended)

To run the system in an isolated container:
```bash
docker-compose up --build -d
```

## Data Sources

- **OpenAQ API:** Sensor network and real-time station-level data.
- **Open-Meteo Weather API:** Meteorological forecasting.
- **Open-Meteo Air Quality API:** Historical time-series and background city pollution levels.
