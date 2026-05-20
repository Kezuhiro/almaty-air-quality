from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from db.database import read_sql_query, append_data


DAILY_TABLE = "daily_features"
HOURLY_TABLE = "hourly_features"

DAILY_RAW_COLUMNS = [
    "date",
    "pm25",
    "pm10",
    "so2",
    "co",
    "temp_mean",
    "temp_min",
    "temp_max",
    "precip",
    "wind_speed_max",
    "wind_speed_mean",
    "wind_dir",
    "pressure",
]

DAILY_TABLE_COLUMNS = [
    "date",
    "pm25",
    "pm10",
    "so2",
    "co",
    "temp_mean",
    "temp_min",
    "temp_max",
    "precip",
    "wind_speed_max",
    "wind_speed_mean",
    "wind_dir",
    "pressure",
    "month",
    "day_of_week",
    "month_sin",
    "month_cos",
    "is_heating_season",
    "pm25_lag1",
    "pm25_lag3",
    "pm25_lag7",
    "pm10_lag1",
    "pm25_roll7_mean",
    "pm25_roll7_std",
    "wind_dir_sin",
    "wind_dir_cos",
    "wind_from_mountains",
    "heating_intensity",
    "is_stagnation",
    "ventilation",
    "inversion_potential",
]

HOURLY_TABLE_COLUMNS = [
    "datetime",
    "temp",
    "precip",
    "wind_speed",
    "pressure",
    "pm10",
    "pm25",
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
]


def _quoted_columns(columns: list[str]) -> str:
    return ", ".join(f'"{column}"' for column in columns)


class DataManager:
    def __init__(self):
        self.daily_table = DAILY_TABLE
        self.hourly_table = HOURLY_TABLE

    def _get_session(self):
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        session.mount("https://", HTTPAdapter(max_retries=retries))
        return session

    def _load_daily_dataset(self, limit: int = 14) -> pd.DataFrame:
        query = f"""
            SELECT {_quoted_columns(DAILY_TABLE_COLUMNS)}
            FROM {self.daily_table}
            ORDER BY date DESC
            LIMIT {limit}
        """
        df = read_sql_query(query, parse_dates=["date"])
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        return df

    def _load_hourly_dataset(self, limit: int = 72) -> pd.DataFrame:
        query = f"""
            SELECT {_quoted_columns(HOURLY_TABLE_COLUMNS)}
            FROM {self.hourly_table}
            ORDER BY "datetime" DESC
            LIMIT {limit}
        """
        df = read_sql_query(query, parse_dates=["datetime"])
        if not df.empty:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.sort_values("datetime").reset_index(drop=True)
        return df

    def _store_daily_dataset(self, dataframe: pd.DataFrame) -> None:
        df_to_store = dataframe[DAILY_TABLE_COLUMNS].copy()
        df_to_store["date"] = pd.to_datetime(df_to_store["date"]).dt.date
        df_to_store = df_to_store.drop_duplicates(subset=["date"], keep="last").sort_values("date")
        append_data(self.daily_table, df_to_store)

    def _store_hourly_dataset(self, dataframe: pd.DataFrame) -> None:
        df_to_store = dataframe[HOURLY_TABLE_COLUMNS].copy()
        df_to_store["datetime"] = pd.to_datetime(df_to_store["datetime"])
        if df_to_store["datetime"].dt.tz is not None:
            df_to_store["datetime"] = df_to_store["datetime"].dt.tz_localize(None)
        df_to_store = df_to_store.drop_duplicates(subset=["datetime"], keep="last").sort_values("datetime")
        append_data(self.hourly_table, df_to_store)

    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        df["month"] = df["date"].dt.month
        df["day_of_week"] = df["date"].dt.dayofweek
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["is_heating_season"] = df["month"].apply(lambda value: 1 if value >= 10 or value <= 4 else 0)

        df["pm25_lag1"] = df["pm25"].shift(1)
        df["pm25_lag3"] = df["pm25"].shift(3)
        df["pm25_lag7"] = df["pm25"].shift(7)
        df["pm10_lag1"] = df["pm10"].shift(1)

        rolling = df["pm25"].shift(1).rolling(window=7)
        df["pm25_roll7_mean"] = rolling.mean()
        df["pm25_roll7_std"] = rolling.std()

        if "wind_dir" in df.columns:
            df["wind_dir_sin"] = np.sin(2 * np.pi * df["wind_dir"] / 360)
            df["wind_dir_cos"] = np.cos(2 * np.pi * df["wind_dir"] / 360)
            df["wind_from_mountains"] = df["wind_dir"].between(135, 225).astype(int)

        df["heating_intensity"] = df["temp_mean"].apply(lambda value: max(0, 18 - value))
        df["is_stagnation"] = ((df["wind_speed_mean"] < 2.0) & (df["precip"] < 0.1)).astype(int)
        df["ventilation"] = df["wind_speed_mean"] * df["temp_max"]
        df["inversion_potential"] = df["pressure"] * (1 / (df["wind_speed_mean"] + 0.1))

        return df

    def update_dataset(self) -> bool:
        df_old = self._load_daily_dataset()
        if df_old.empty:
            print("Таблица daily_features пуста. Сначала выполните начальную загрузку в PostgreSQL.")
            return False

        df_old["date"] = pd.to_datetime(df_old["date"]).dt.date
        last_date = df_old["date"].max()
        yesterday = date.today() - timedelta(days=1)

        if last_date >= yesterday:
            print(f"Суточные данные в PostgreSQL уже актуальны: {last_date}")
            return True

        start_str = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        end_str = yesterday.strftime("%Y-%m-%d")
        session = self._get_session()

        try:
            weather_url = "https://api.open-meteo.com/v1/forecast"
            weather_params = {
                "latitude": 43.25,
                "longitude": 76.95,
                "start_date": start_str,
                "end_date": end_str,
                "daily": [
                    "temperature_2m_mean",
                    "temperature_2m_min",
                    "temperature_2m_max",
                    "precipitation_sum",
                    "wind_speed_10m_max",
                    "wind_direction_10m_dominant",
                ],
                "hourly": ["surface_pressure"],
                "timezone": "Asia/Almaty",
            }
            weather_response = session.get(weather_url, params=weather_params, timeout=10)
            weather_response.raise_for_status()
            weather_data = weather_response.json()
            weather_daily = weather_data["daily"]

            pressure_df = pd.DataFrame(
                {
                    "time": pd.to_datetime(weather_data["hourly"]["time"]),
                    "pressure": weather_data["hourly"]["surface_pressure"],
                }
            )
            pressure_df["date"] = pressure_df["time"].dt.date
            daily_pressure = pressure_df.groupby("date")["pressure"].mean().values

            air_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
            air_params = {
                "latitude": 43.25,
                "longitude": 76.95,
                "start_date": start_str,
                "end_date": end_str,
                "hourly": ["pm2_5", "pm10", "sulphur_dioxide", "carbon_monoxide"],
                "timezone": "Asia/Almaty",
            }
            air_response = session.get(air_url, params=air_params, timeout=10)
            air_response.raise_for_status()
            air_hourly = air_response.json()["hourly"]

            air_df = pd.DataFrame(air_hourly)
            air_df["date"] = pd.to_datetime(air_df["time"]).dt.date
            air_daily = (
                air_df.groupby("date")
                .agg(
                    {
                        "pm2_5": "mean",
                        "pm10": "mean",
                        "sulphur_dioxide": "mean",
                        "carbon_monoxide": "mean",
                    }
                )
                .reset_index()
            )

            new_raw = pd.DataFrame(
                {
                    "date": pd.to_datetime(weather_daily["time"]).date,
                    "temp_mean": weather_daily["temperature_2m_mean"],
                    "temp_min": weather_daily["temperature_2m_min"],
                    "temp_max": weather_daily["temperature_2m_max"],
                    "precip": weather_daily["precipitation_sum"],
                    "wind_speed_max": weather_daily["wind_speed_10m_max"],
                    "wind_speed_mean": np.array(weather_daily["wind_speed_10m_max"]) * 0.7,
                    "wind_dir": weather_daily["wind_direction_10m_dominant"],
                    "pressure": daily_pressure,
                    "pm25": air_daily["pm2_5"],
                    "pm10": air_daily["pm10"],
                    "so2": air_daily["sulphur_dioxide"],
                    "co": air_daily["carbon_monoxide"],
                }
            )

            combined_raw = pd.concat([df_old[DAILY_RAW_COLUMNS], new_raw], ignore_index=True)
            final_df = self._calculate_features(combined_raw)
            self._store_daily_dataset(final_df)
            print(f"Суточный датасет в PostgreSQL обновлен до {yesterday}")
            return True
        except Exception as exc:
            print(f"Ошибка обновления суточного датасета в PostgreSQL: {exc}")
            return False

    def _calculate_hourly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime")

        if "wind_dir" in df.columns:
            df["wind_dir_sin"] = np.sin(2 * np.pi * df["wind_dir"] / 360)
            df["wind_dir_cos"] = np.cos(2 * np.pi * df["wind_dir"] / 360)

        df["hour"] = df["datetime"].dt.hour
        df["month"] = df["datetime"].dt.month
        df["day_of_week"] = df["datetime"].dt.dayofweek

        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["is_heating_season"] = df["month"].apply(lambda value: 1 if value >= 10 or value <= 4 else 0)

        df["pm25_lag1"] = df["pm25"].shift(1)
        df["pm25_lag2"] = df["pm25"].shift(2)
        df["pm25_lag3"] = df["pm25"].shift(3)
        df["pm25_lag24"] = df["pm25"].shift(24)

        df["pm25_lag48"] = df["pm25"].shift(48)
        df["pm25_lag72"] = df["pm25"].shift(72)
        df["pm25_wind_interaction"] = df["pm25_lag1"] * df["wind_speed"]

        rolling24 = df["pm25"].shift(1).rolling(window=24)
        df["pm25_roll24_mean"] = rolling24.mean()
        df["pm25_roll24_std"] = rolling24.std()

        return df

    def update_hourly_dataset(self) -> bool:
        df_old = self._load_hourly_dataset()
        if df_old.empty:
            print("Таблица hourly_features пуста. Сначала выполните начальную загрузку в PostgreSQL.")
            return False

        df_old["datetime"] = pd.to_datetime(df_old["datetime"])
        last_date = df_old["datetime"].max().date()
        yesterday = date.today() - timedelta(days=1)

        if last_date >= yesterday:
            print(f"Часовые данные в PostgreSQL уже актуальны: {last_date}")
            return True

        start_str = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        end_str = yesterday.strftime("%Y-%m-%d")
        session = self._get_session()

        try:
            weather_url = "https://api.open-meteo.com/v1/forecast"
            weather_params = {
                "latitude": 43.25,
                "longitude": 76.95,
                "start_date": start_str,
                "end_date": end_str,
                "hourly": [
                    "temperature_2m",
                    "precipitation",
                    "wind_speed_10m",
                    "surface_pressure",
                    "wind_direction_10m",
                ],
                "timezone": "Asia/Almaty",
            }
            weather_response = session.get(weather_url, params=weather_params, timeout=10)
            weather_response.raise_for_status()
            weather_hourly = weather_response.json()["hourly"]

            air_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
            air_params = {
                "latitude": 43.25,
                "longitude": 76.95,
                "start_date": start_str,
                "end_date": end_str,
                "hourly": ["pm10", "pm2_5", "carbon_monoxide", "sulphur_dioxide"],
                "timezone": "Asia/Almaty",
            }
            air_response = session.get(air_url, params=air_params, timeout=10)
            air_response.raise_for_status()
            air_hourly = air_response.json()["hourly"]

            new_raw = pd.DataFrame(
                {
                    "datetime": pd.to_datetime(weather_hourly["time"]),
                    "temp": weather_hourly["temperature_2m"],
                    "precip": weather_hourly["precipitation"],
                    "wind_speed": np.array(weather_hourly["wind_speed_10m"]) * 0.7,
                    "wind_dir": weather_hourly["wind_direction_10m"],
                    "pressure": weather_hourly["surface_pressure"],
                    "pm10": air_hourly["pm10"],
                    "pm25": air_hourly["pm2_5"],
                    "co": air_hourly["carbon_monoxide"],
                    "so2": air_hourly["sulphur_dioxide"],
                }
            )

            new_features = self._calculate_hourly_features(new_raw)
            combined_df = pd.concat([df_old, new_features], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=["datetime"], keep="last")
            final_df = self._calculate_hourly_features(combined_df)
            final_df = final_df[HOURLY_TABLE_COLUMNS]
            self._store_hourly_dataset(final_df)
            print(f"Часовой датасет в PostgreSQL обновлен до {yesterday}")
            return True
        except Exception as exc:
            print(f"Ошибка обновления часового датасета в PostgreSQL: {exc}")
            return False

    def get_last_features(self) -> dict:
        query = f"""
            SELECT {_quoted_columns(DAILY_TABLE_COLUMNS)}
            FROM {self.daily_table}
            ORDER BY date DESC
            LIMIT 1
        """
        df = read_sql_query(query, parse_dates=["date"])
        if df.empty:
            return {}
        return df.iloc[0].to_dict()


data_manager = DataManager()
