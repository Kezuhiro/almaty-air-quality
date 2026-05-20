# Air Quality Monitoring and Forecasting for Almaty

This project is an advanced, production-ready web platform designed to monitor, simulate, and forecast air quality in Almaty. By combining real-time IoT pollution metrics with historical atmospheric and meteorological parameters, the platform delivers high-fidelity city-wide reports, localized station-level projections, and hypothetical scenario modeling. 

The application has transitioned from a file-based storage architecture (CSV files) to a structured **PostgreSQL Relational Database** with **PostGIS** geo-spatial extensions, managed through a modern **SQLAlchemy 2.0** ORM.

---

## Key Features

* **Relational Database Engine**: A centralized database architecture tracking real-time sensor metrics, spatial graph relationships, and engineered feature tables, eliminating stale CSV files.
* **Modern Web Interface (5 Core Pages)**:
  * **Home (Dashboard)**: Real-time city-wide average PM2.5, instant weather widgets, pollution trend charts, and immediate next-day recommendations.
  * **Forecast (ML Predictions)**: A consolidated forecasting deck hosting the daily city baseline, the Spatio-Temporal GNN station forecast, and a **newly integrated autoregressive 24-hour XGBoost timeline**.
  * **History (Data Analytics)**: Comprehensive historical charting for comparative analysis between weather indicators (temperature, wind speed, pressure, humidity) and PM2.5/PM10 levels.
  * **Simulation (3D Scenario Emulator)**: An interactive 3D model designed to simulate pollution dispersion across 24 stations. Users can tune mock temperature, wind speed/direction, humidity, and timeframe parameters to visualize wind vector vectors and spatial migration.
  * **About (System Info)**: Project documentation, math formulas (Haversine distance, graph adjacency), model features, and target information.
* **Tri-Model ML Forecasting Engine**:
  * **CatBoost**: Daily city-wide average PM2.5 forecasting.
  * **Spatio-Temporal Graph Convolutional Network (STGCN)**: Sub-daily spatial forecasting across 24 anchor stations using graph edge weighting.
  * **XGBoost (Hourly 24-Hour)**: A new autoregressive model predicting sequential hourly PM2.5 values using rolling statistics, lags, and wind vector interactions.
* **Nearest-Neighbor Spatial Imputation**: An online distance-weighted fallback system using the Haversine formula to instantly recover sensor dropouts when physical IoT units lose connection.

---

## Database Schema

The database architecture (managed in `app/db/`) utilizes PostgreSQL to persist time-series data and spatial relations.

```mermaid
erDiagram
    stations ||--o{ measurements : "records"
    stations ||--o{ station_distances : "source"
    stations ||--o{ station_distances : "target"
    stations ||--o{ forecasts : "predicts"
    
    stations {
        int id PK
        string name
        float lat
        float lon
    }
    station_distances {
        int source_id PK, FK
        int target_id PK, FK
        float distance
    }
    measurements {
        int id PK
        int station_id FK
        datetime timestamp
        float pm25
        float pm10
        boolean is_imputed
    }
    weather {
        int id PK
        datetime timestamp
        float wind_speed
        float wind_dir
        float temp
        float humidity
        float pressure
    }
    forecasts {
        int id PK
        datetime created_at
        datetime target_timestamp
        string model_name
        int station_id FK
        float pm25_pred
    }
    daily_features {
        int id PK
        date date
        float pm25
        float pm10
        float temp_mean
        float temp_min
        float temp_max
        float precip
        float wind_speed_max
        float wind_speed_mean
        float wind_dir
        float pressure
        float pm25_lag1
        float pm25_lag3
        float pm25_lag7
        float pm25_roll7_mean
        float pm25_roll7_std
        float ventilation
        float inversion_potential
        int is_heating_season
    }
    hourly_features {
        int id PK
        datetime datetime
        float temp
        float precip
        float wind_speed
        float pressure
        float pm25
        float pm10
        float co
        float so2
        float pm25_lag1
        float pm25_lag2
        float pm25_lag3
        float pm25_lag24
        float pm25_lag48
        float pm25_lag72
        float pm25_roll24_mean
        float pm25_roll24_std
        float pm25_wind_interaction
    }
```

---

## Machine Learning Frameworks

The platform houses three pre-trained machine learning systems designed for different temporal and spatial granularities:

### 1. Daily Baseline Forecast (CatBoost)
Predicts the overall city-wide average PM2.5 level for the next calendar day.
* **Feature Engineering**: Incorporates temporal features (month, day-of-week, sinusoids), calendar flags (heating season vs. summer), lags (1-day, 3-day, 7-day average), and physics-informed weather combinations:
  * **Thermal Inversion Potential**: Estimating atmospheric capping based on pressure and wind speed: $I_p = \text{Pressure} \times \frac{1}{\text{Wind Speed} + 0.1}$
  * **Ventilation Index**: Atmospheric clearance potential calculated as: $V = \text{Wind Speed Mean} \times \text{Temp Max}$
  * **Mountain Wind Interactions**: Capturing nocturnal downhill breeze patterns blowing from the southern Tian Shan range (wind directions between $135^\circ$ and $225^\circ$).

### 2. Spatio-Temporal Graph Neural Network (STGCN)
Performs short-term spatial forecast across 24 distributed anchor stations in the Almaty sensor mesh.
* **Mathematical Foundation**: Maps the city sensors as nodes in a graph $G = (V, E, W)$.
* **Adjacency Matrix ($W$)**: Defines spatial connectivity utilizing geographical distances computed via the Haversine formula, applying a thresholding kernel to create localized spatial influences:
  $$W_{ij} = \exp\left(-\frac{\text{haversine}(i, j)^2}{\sigma^2}\right)$$
* **Architecture**: Combines spatial graph convolutions (Chebyshev GCNs) with 1D temporal gated convolution blocks (GLUs) to forecast smog migration and local dispersion trends.

### 3. Hourly Autoregressive Forecast (XGBoost)
Predicts sequential hourly PM2.5 values for the **next 24 hours**.
* **Methodology**: Operates as an autoregressive sliding window model. The prediction at step $t$ updates the lag values ($t-1, t-2, \dots$) iteratively to project values up to 24 steps forward.
* **Engineered Lags**: Utilizes extensive short-term and long-term time-series dependencies:
  * **Short-Term Lags**: `pm25_lag1`, `pm25_lag2`, `pm25_lag3` (capturing immediate local autocorrelation).
  * **Diurnal Lags**: `pm25_lag24`, `pm25_lag48`, `pm25_lag72` (capturing daily cyclical patterns).
  * **Rolling Metrics**: `pm25_roll24_mean`, `pm25_roll24_std` (capturing moving trend windows).
  * **Wind Vector Interaction**: `pm25_lag1 * wind_speed` (capturing weather-dependent dispersal acceleration).

---

## Project Structure

```text
├── app/
│   ├── main.py                     # FastAPI core router and startup initialization
│   ├── db/
│   │   ├── database.py             # SQLAlchemy engine connection and bulk write utilities
│   │   └── models.py               # Declarative schema classes mapping PostgreSQL tables
│   ├── services/
│   │   ├── catboost_service.py     # CatBoost daily inference wrapper
│   │   ├── stgcn_service.py        # STGCN spatial graph engine and heatmap generator
│   │   ├── xgb_hourly_service.py   # Autoregressive XGBoost 24h hourly forecaster
│   │   ├── data_manager.py         # DB orchestrator, fetching weather & API metrics
│   │   ├── fetcher.py              # OpenAQ and Open-Meteo external REST APIs interface
│   │   ├── imputer.py              # Spatial nearest-neighbor sensor value restorer
│   │   └── history_service.py      # Aggregator preparing history datasets for UI charts
│   ├── templates/                  # Jinja2 templates (base.html + 5 page layouts)
│   │   ├── base.html
│   │   ├── home.html
│   │   ├── forecast.html
│   │   ├── history.html
│   │   ├── simulation.html
│   │   └── about.html
│   └── static/                     # CSS, frontend JS assets, charts configurations
│
├── data/                           # Local seed raw tables (used only for DB seeding)
│   ├── almaty_stations_coordinates.csv
│   ├── almaty_pm25_matrix.csv
│   └── train_hourly_complete.csv
│
├── models/                         # Pre-trained model artifacts loaded into memory
│   ├── catboost_daily/
│   │   └── catboost_final.cbm
│   ├── stgcn_24h/
│   │   └── artifacts/
│   │       ├── stgcn_almaty.pth
│   │       ├── adj_matrix.npy
│   │       └── pm25_scaler.pkl
│   └── xgb_hourly/
│       └── xgb_pipeline.joblib
│
├── scripts/
│   └── seed_db.py                  # CLI utility seeding PostgreSQL tables with historical CSV data
│
├── Dockerfile                      # Multistage Python application container description
├── docker-compose.yml              # PostgreSQL PostGIS DB container + Python API container
├── requirements.txt                # Unified dependency list
└── README.md                       # Documentation
```

---

## Installation and Setup

### Prerequisites
* **Docker** & **Docker Compose** (Highly Recommended)
* Python `3.10` or `3.11` (if running locally without Docker; newer versions may fail compiling `torch` or `catboost` wheels).

### 1. Docker Setup (Recommended)
This launches a **PostGIS/PostgreSQL** container along with the **FastAPI Web Service** in isolated Docker network segments.

Before running the application, you need to set up your environment variables:

1. **Copy the environment template file**:
   ```bash
   cp .env.example .env
   ```

1. **Configure Environment Variables**:
   Create a `.env` file in the root directory:
  ```env
  # PostgreSQL Database Configuration
  DB_USER=postgres
  DB_PASSWORD=yoursupersecretpassword
  DB_HOST=db
  DB_PORT=5432
  DB_NAME=air_quality

  # Third-Party API Keys
  OPENAQ_API_KEY=your_openaq_api_key
   ```

2. **Spin Up Containers**:
   Build and start both services:
   ```bash
   docker-compose up --build -d
   ```

3. **Seed Database Tables**:
   Before the models can predict or display dashboards, seed the PostgreSQL tables with historical information:
   ```bash
   docker-compose exec air_quality_api python scripts/seed_db.py
   ```
   *This seeds station coordinates, computes the distance-adjacency matrix, loads spatial matrices, and stores historical daily/hourly features directly into the PostgreSQL database.*

---

### 2. Local Setup (Without Docker)
If you prefer running the Python server locally, make sure you have access to a running PostgreSQL database.

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Export Database URL**:
   Ensure the synchronization environment variable is available in your shell:
   ```powershell
   # Windows PowerShell
   $env:SYNC_DATABASE_URL="postgresql+psycopg2://postgres:yoursupersecretpassword@localhost:5432/air_quality"
   $env:OPENAQ_API_KEY="your_openaq_api_key_here"
   ```
   ```bash
   # Unix/macOS
   export SYNC_DATABASE_URL="postgresql+psycopg2://postgres:yoursupersecretpassword@localhost:5432/air_quality"
   export OPENAQ_API_KEY="your_openaq_api_key_here"
   ```

3. **Run Seeding Script**:
   ```bash
   python scripts/seed_db.py
   ```

4. **Launch the FastAPI Server**:
   ```bash
   cd app
   python main.py
   ```
   The platform will be available locally at `http://127.0.0.1:8000`.

---

## API Endpoints Reference

### Web UI Routes
* `GET /` — Renders `home.html` (Dashboard).
* `GET /forecast` — Renders `forecast.html` (Baseline CatBoost, GNN spatial graph, and XGBoost 24h predictions).
* `GET /history` — Renders `history.html` (Analytics and trend charts).
* `GET /simulation` — Renders `simulation.html` (3D interactive simulator).
* `GET /about` — Renders `about.html` (Methodology and specs).

### Data REST APIs
* `GET /map` — Returns dynamic Folium spatial heatmap layers.
* `POST /api/simulate` — Solves spatial smog migration matrices based on custom climate payloads.
* `GET /api/forecast/hourly-xgb` — Runs the 24-step autoregressive XGBoost model using loaded database warm-ups and current forecasts.
* `GET /api/predict/{model_name}` — Evaluates model readiness checks (e.g. `stgcn`, `catboost`, `xgb_hourly`).

---

## External Data Integrations
The application pulls raw external data via asynchronous HTTP fetchers:
* **OpenAQ API**: Gathers real-time IoT PM2.5 and PM10 measurements from physical monitoring stations in Almaty.
* **Open-Meteo Weather API**: Retrieves hourly wind speed, wind direction, temperature, surface pressure, and precipitation.
* **Open-Meteo Air Quality API**: Fetches raw CO, SO2, and PM2.5 background references to support model feature engines.
