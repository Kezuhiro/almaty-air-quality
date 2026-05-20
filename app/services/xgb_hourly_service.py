import os
from datetime import datetime

import httpx
import joblib
import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgb_hourly", "xgb_pipeline.joblib")


class XGBHourlyService:
    def __init__(self):
        self.model = None
        if os.path.exists(MODEL_PATH):
            try:
                self.model = joblib.load(MODEL_PATH)
                print(f"XGBoost Hourly model loaded from: {MODEL_PATH}")
            except Exception as exc:
                print(f"XGBoost Hourly load error: {exc}")

        self.feature_columns = [
            "temp",
            "precip",
            "wind_speed",
            "pressure",
            "pm10",
            "co",
            "so2",
            "day_of_week",
            "is_heating_season",
            "hour_sin",
            "hour_cos",
            "month_sin",
            "month_cos",
            "wind_dir_sin",
            "wind_dir_cos",
            "pm25_lag1",
            "pm25_lag2",
            "pm25_lag3",
            "pm25_lag24",
            "pm25_roll24_mean",
            "pm25_roll24_std",
            "pm25_lag48",
            "pm25_lag72",
            "pm25_wind_interaction",
        ]

    async def _fetch_future_weather_24h(self) -> pd.DataFrame:
        url_weather = "https://api.open-meteo.com/v1/forecast"
        url_air = "https://air-quality-api.open-meteo.com/v1/air-quality"
        params = {
            "latitude": 43.25,
            "longitude": 76.95,
            "forecast_days": 2,
            "timezone": "Asia/Almaty",
        }

        async with httpx.AsyncClient() as client:
            weather_response = await client.get(
                url_weather,
                params={
                    **params,
                    "hourly": [
                        "temperature_2m",
                        "precipitation",
                        "wind_speed_10m",
                        "surface_pressure",
                        "wind_direction_10m",
                    ],
                },
            )
            air_response = await client.get(
                url_air,
                params={
                    **params,
                    "hourly": ["pm10", "carbon_monoxide", "sulphur_dioxide"],
                },
            )
            weather_response.raise_for_status()
            air_response.raise_for_status()

            weather_data = weather_response.json()["hourly"]
            air_data = air_response.json()["hourly"]

            df = pd.DataFrame(
                {
                    "datetime": pd.to_datetime(weather_data["time"]),
                    "temp": weather_data["temperature_2m"],
                    "precip": weather_data["precipitation"],
                    "wind_speed": np.array(weather_data["wind_speed_10m"]) * 0.7,
                    "wind_dir": weather_data["wind_direction_10m"],
                    "pressure": weather_data["surface_pressure"],
                    "pm10": air_data["pm10"],
                    "co": air_data["carbon_monoxide"],
                    "so2": air_data["sulphur_dioxide"],
                }
            )

        now = datetime.now()
        return df[df["datetime"] > now].sort_values("datetime").head(24).reset_index(drop=True)

    async def predict_next_24h(self, df_history: pd.DataFrame) -> list[dict]:
        if self.model is None:
            return []

        if df_history.empty or len(df_history) < 72:
            raise ValueError("For XGB Hourly at least 72 hours of history are required.")

        df_future = await self._fetch_future_weather_24h()
        if df_future.empty:
            raise ValueError("No future weather data returned for XGB Hourly forecast.")

        df_future["hour"] = df_future["datetime"].dt.hour
        df_future["month"] = df_future["datetime"].dt.month
        df_future["day_of_week"] = df_future["datetime"].dt.dayofweek
        df_future["is_heating_season"] = df_future["month"].apply(lambda month: 1 if month >= 10 or month <= 4 else 0)
        df_future["hour_sin"] = np.sin(2 * np.pi * df_future["hour"] / 24)
        df_future["hour_cos"] = np.cos(2 * np.pi * df_future["hour"] / 24)
        df_future["month_sin"] = np.sin(2 * np.pi * df_future["month"] / 12)
        df_future["month_cos"] = np.cos(2 * np.pi * df_future["month"] / 12)
        df_future["wind_dir_sin"] = np.sin(2 * np.pi * df_future["wind_dir"] / 360)
        df_future["wind_dir_cos"] = np.cos(2 * np.pi * df_future["wind_dir"] / 360)
        df_future["pm25"] = np.nan

        history_columns = [
            "datetime",
            "temp",
            "precip",
            "wind_speed",
            "pressure",
            "pm10",
            "co",
            "so2",
            "day_of_week",
            "is_heating_season",
            "hour_sin",
            "hour_cos",
            "month_sin",
            "month_cos",
            "wind_dir_sin",
            "wind_dir_cos",
            "pm25",
        ]
        df_history = df_history.sort_values("datetime").tail(72).reset_index(drop=True)
        df_hist_clean = df_history[history_columns].copy()
        df_total = pd.concat([df_hist_clean, df_future], ignore_index=True).sort_values("datetime").reset_index(drop=True)

        history_len = len(df_hist_clean)
        predictions: list[dict] = []

        for index in range(history_len, len(df_total)):
            df_total.loc[index, "pm25_lag1"] = df_total.loc[index - 1, "pm25"]
            df_total.loc[index, "pm25_lag2"] = df_total.loc[index - 2, "pm25"]
            df_total.loc[index, "pm25_lag3"] = df_total.loc[index - 3, "pm25"]
            df_total.loc[index, "pm25_lag24"] = df_total.loc[index - 24, "pm25"]
            df_total.loc[index, "pm25_lag48"] = df_total.loc[index - 48, "pm25"]
            df_total.loc[index, "pm25_lag72"] = df_total.loc[index - 72, "pm25"]
            df_total.loc[index, "pm25_wind_interaction"] = df_total.loc[index, "pm25_lag1"] * df_total.loc[index, "wind_speed"]

            rolling_window = df_total.loc[index - 23:index - 1, "pm25"].tolist() + [df_total.loc[index, "pm25_lag1"]]
            df_total.loc[index, "pm25_roll24_mean"] = np.mean(rolling_window)
            df_total.loc[index, "pm25_roll24_std"] = np.std(rolling_window) if len(rolling_window) > 1 else 0.0

            features = df_total.loc[[index], self.feature_columns]
            prediction = float(self.model.predict(features)[0])
            prediction = max(0.0, prediction)
            df_total.loc[index, "pm25"] = prediction

            predictions.append(
                {
                    "time": df_total.loc[index, "datetime"].strftime("%H:%M"),
                    "pm25": round(prediction, 1),
                }
            )

        return predictions


xgb_hourly_service = XGBHourlyService()
