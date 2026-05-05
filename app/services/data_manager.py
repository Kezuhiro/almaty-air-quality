import os
import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, timedelta, date

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_PATH = os.path.join(BASE_DIR, "data", "train_dataset_full.csv")
HOURLY_CSV_PATH = os.path.join(BASE_DIR, "data", "train_hourly_complete.csv")

class DataManager:
    def __init__(self):
        self.csv_path = CSV_PATH
        self.hourly_csv_path = HOURLY_CSV_PATH

    def _get_session(self):
        """Создает HTTP-сессию с таймаутами и автоматическими ретраями"""
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))
        return session

    
    # ЛОГИКА СУТОЧНЫХ ДАННЫХ (CatBoost Daily)
  

    def _calculate_features(self, df):
        """Полный расчет всех признаков для суточного CatBoost"""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        df['is_heating_season'] = df['month'].apply(lambda x: 1 if x >= 10 or x <= 4 else 0)

        df['pm25_lag1'] = df['pm25'].shift(1)
        df['pm25_lag3'] = df['pm25'].shift(3)
        df['pm25_lag7'] = df['pm25'].shift(7)
        df['pm10_lag1'] = df['pm10'].shift(1)
        
        rolling = df['pm25'].shift(1).rolling(window=7)
        df['pm25_roll7_mean'] = rolling.mean()
        df['pm25_roll7_std'] = rolling.std()

        if 'wind_dir' in df.columns:
            df['wind_dir_sin'] = np.sin(2 * np.pi * df['wind_dir'] / 360)
            df['wind_dir_cos'] = np.cos(2 * np.pi * df['wind_dir'] / 360)
            df['wind_from_mountains'] = df['wind_dir'].between(135, 225).astype(int)
        
        df['heating_intensity'] = df['temp_mean'].apply(lambda x: max(0, 18 - x))
        df['is_stagnation'] = ((df['wind_speed_mean'] < 2.0) & (df['precip'] < 0.1)).astype(int)
        df['ventilation'] = df['wind_speed_mean'] * df['temp_max']
        df['inversion_potential'] = df['pressure'] * (1 / (df['wind_speed_mean'] + 0.1))

        return df

    def update_dataset(self):
        """Скачивает сырые данные и обновляет суточный CSV"""
        if not os.path.exists(self.csv_path): 
            print("Файл суточного датасета не найден.")
            return False

        df_old = pd.read_csv(self.csv_path)
        df_old['date'] = pd.to_datetime(df_old['date']).dt.date
        
        last_date = df_old['date'].max()
        yesterday = date.today() - timedelta(days=1)

        if last_date >= yesterday:
            print(f"Суточные данные актуальны: {last_date}")
            return True

        start_str = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        end_str = yesterday.strftime("%Y-%m-%d")
        session = self._get_session()

        try:
            weather_url = "https://api.open-meteo.com/v1/forecast"
            weather_params = {
                "latitude": 43.25, "longitude": 76.95,
                "start_date": start_str, "end_date": end_str,
                "daily": ["temperature_2m_mean", "temperature_2m_min", "temperature_2m_max", 
                          "precipitation_sum", "wind_speed_10m_max", "wind_direction_10m_dominant"],
                "hourly": ["surface_pressure"],
                "timezone": "Asia/Almaty"
            }
            w_req = session.get(weather_url, params=weather_params, timeout=10)
            w_req.raise_for_status()
            w_data = w_req.json()
            w_daily = w_data['daily']
            
            pressure_df = pd.DataFrame({
                "time": pd.to_datetime(w_data['hourly']['time']),
                "pressure": w_data['hourly']['surface_pressure']
            })
            pressure_df['date'] = pressure_df['time'].dt.date
            daily_pressure = pressure_df.groupby('date')['pressure'].mean().values
            
            air_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
            air_params = {
                "latitude": 43.25, "longitude": 76.95,
                "start_date": start_str, "end_date": end_str,
                "hourly": ["pm2_5", "pm10", "sulphur_dioxide", "carbon_monoxide"],
                "timezone": "Asia/Almaty"
            }
            a_req = session.get(air_url, params=air_params, timeout=10)
            a_req.raise_for_status()
            a_resp = a_req.json()['hourly']
            
            air_df = pd.DataFrame(a_resp)
            air_df['date'] = pd.to_datetime(air_df['time']).dt.date
            air_daily = air_df.groupby('date').agg({
                'pm2_5': 'mean', 'pm10': 'mean', 
                'sulphur_dioxide': 'mean', 'carbon_monoxide': 'mean'
            }).reset_index()

            new_raw = pd.DataFrame({
                "date": pd.to_datetime(w_daily["time"]).date, 
                "temp_mean": w_daily["temperature_2m_mean"],
                "temp_min": w_daily["temperature_2m_min"],
                "temp_max": w_daily["temperature_2m_max"],
                "precip": w_daily["precipitation_sum"],
                "wind_speed_max": w_daily["wind_speed_10m_max"],
                "wind_speed_mean": np.array(w_daily["wind_speed_10m_max"]) * 0.7,
                "wind_dir": w_daily["wind_direction_10m_dominant"],
                "pressure": daily_pressure,  
                "pm25": air_daily['pm2_5'],
                "pm10": air_daily['pm10'],
                "so2": air_daily['sulphur_dioxide'],
                "co": air_daily['carbon_monoxide']
            })

            combined_raw = pd.concat([df_old[['date', 'pm25', 'pm10', 'so2', 'co', 'temp_mean', 'temp_min', 
                                            'temp_max', 'precip', 'wind_speed_max', 'wind_speed_mean', 
                                            'wind_dir', 'pressure']], new_raw], ignore_index=True)
            
            final_df = self._calculate_features(combined_raw)
            final_df.to_csv(self.csv_path, index=False)
            print(f"Суточный CSV успешно обновлен до {yesterday}")
            return True

        except Exception as e:
            print(f"Ошибка обновления суточного CSV: {e}")
            return False


   
    # ЛОГИКА ЧАСОВЫХ ДАННЫХ (Baseline/STGCN)


    def _calculate_hourly_features(self, df):
        """Полный пересчет всех признаков для часовых моделей"""
        df = df.copy()
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime')

        # Если есть сырой ветер (из новых данных API), считаем sin/cos
        if 'wind_dir' in df.columns:
            df['wind_dir_sin'] = np.sin(2 * np.pi * df['wind_dir'] / 360)
            df['wind_dir_cos'] = np.cos(2 * np.pi * df['wind_dir'] / 360)

        # Календарные фичи
        df['hour'] = df['datetime'].dt.hour
        df['month'] = df['datetime'].dt.month
        df['day_of_week'] = df['datetime'].dt.dayofweek

        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['is_heating_season'] = df['month'].apply(lambda x: 1 if x >= 10 or x <= 4 else 0)

        # Временные лаги
        df['pm25_lag1'] = df['pm25'].shift(1)
        df['pm25_lag2'] = df['pm25'].shift(2)
        df['pm25_lag3'] = df['pm25'].shift(3)
        df['pm25_lag24'] = df['pm25'].shift(24)

        # Скользящие окна за последние 24 часа
        rolling24 = df['pm25'].shift(1).rolling(window=24)
        df['pm25_roll24_mean'] = rolling24.mean()
        df['pm25_roll24_std'] = rolling24.std()

        return df

    def update_hourly_dataset(self):
        """Скачивает сырые данные и обновляет часовой CSV"""
        if not os.path.exists(self.hourly_csv_path): 
            print("Файл часового датасета не найден.")
            return False

        df_old = pd.read_csv(self.hourly_csv_path)
        df_old['datetime'] = pd.to_datetime(df_old['datetime'])
        
        last_date = df_old['datetime'].max().date()
        yesterday = date.today() - timedelta(days=1)

        if last_date >= yesterday:
            print(f"Часовые данные актуальны: {last_date}")
            return True

        start_str = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        end_str = yesterday.strftime("%Y-%m-%d")
        session = self._get_session()

        try:
            # 1. Погода по часам
            weather_url = "https://api.open-meteo.com/v1/forecast"
            weather_params = {
                "latitude": 43.25, "longitude": 76.95,
                "start_date": start_str, "end_date": end_str,
                "hourly": ["temperature_2m", "precipitation", "wind_speed_10m", 
                           "surface_pressure", "wind_direction_10m"],
                "timezone": "Asia/Almaty"
            }
            w_req = session.get(weather_url, params=weather_params, timeout=10)
            w_req.raise_for_status()
            w_hourly = w_req.json()['hourly']

            # 2. Качество воздуха по часам
            air_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
            air_params = {
                "latitude": 43.25, "longitude": 76.95,
                "start_date": start_str, "end_date": end_str,
                "hourly": ["pm10", "pm2_5", "carbon_monoxide", "sulphur_dioxide"],
                "timezone": "Asia/Almaty"
            }
            a_req = session.get(air_url, params=air_params, timeout=10)
            a_req.raise_for_status()
            a_hourly = a_req.json()['hourly']

            # 3. Слияние новых сырых данных
            new_raw = pd.DataFrame({
                "datetime": pd.to_datetime(w_hourly["time"]),
                "temp": w_hourly["temperature_2m"],
                "precip": w_hourly["precipitation"],
                "wind_speed": np.array(w_hourly["wind_speed_10m"]) * 0.7,
                "wind_dir": w_hourly["wind_direction_10m"],
                "pressure": w_hourly["surface_pressure"],
                "pm10": a_hourly["pm10"],
                "pm25": a_hourly["pm2_5"],
                "co": a_hourly["carbon_monoxide"],
                "so2": a_hourly["sulphur_dioxide"]
            })

            # Предварительно считаем sin/cos ветра только для новых данных, 
            # чтобы потом без проблем склеить с old_df, у которого нет wind_dir
            new_features = self._calculate_hourly_features(new_raw)

            # Объединяем старый и новый датафреймы
            combined_df = pd.concat([df_old, new_features], ignore_index=True)

            # Пересчитываем лаги и окна на всем объединенном массиве (для бесшовного стыка)
            final_df = self._calculate_hourly_features(combined_df)

            # 4. Жесткая фильтрация итоговых колонок под заявленный формат
            required_cols = [
                "datetime", "temp", "precip", "wind_speed", "pressure", "pm10", "pm25", "co", "so2",
                "day_of_week", "is_heating_season", "hour_sin", "hour_cos", "month_sin", "month_cos",
                "wind_dir_sin", "wind_dir_cos", "pm25_lag1", "pm25_lag2", "pm25_lag3", "pm25_lag24",
                "pm25_roll24_mean", "pm25_roll24_std"
            ]
            
            final_df = final_df[required_cols]
            final_df.to_csv(self.hourly_csv_path, index=False)
            
            print(f"Часовой CSV успешно обновлен до {yesterday}")
            return True

        except Exception as e:
            print(f"Ошибка обновления часового CSV: {e}")
            return False

    def get_last_features(self):
        """Возвращает последнюю строку со всеми вычисленными суточными признаками"""
        df = pd.read_csv(self.csv_path)
        return df.iloc[-1].to_dict()

data_manager = DataManager()