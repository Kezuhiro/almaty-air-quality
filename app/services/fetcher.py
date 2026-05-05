import requests
import pandas as pd
from services.imputer import SpatialImputer
import time
import math
import httpx
import datetime


class OpenMeteoFetcher:
    def __init__(self, lat=43.25, lon=76.95):
        self.lat = lat
        self.lon = lon
        
        self.cache_ttl = 3600  # 1 час (3600 секунд)
        
        self.cached_weather = None
        self.weather_time = 0
        
        self.cached_history = None
        self.history_time = 0

    async def get_current_weather(self):
        """Получает текущую погоду (с кэшированием на 1 час)"""
        current_time = time.time()
        if self.cached_weather and (current_time - self.weather_time) < self.cache_ttl:
            return self.cached_weather

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "current": ["temperature_2m", "wind_speed_10m"],
            "timezone": "Asia/Almaty"
        }
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(url, params=params, timeout=10.0)
                resp.raise_for_status()
                data = resp.json()
                
                result = {
                    "temp": round(data["current"]["temperature_2m"]),
                    "wind": round(data["current"]["wind_speed_10m"], 1)
                }
                
                # Сохраняем в кэш
                self.cached_weather = result
                self.weather_time = current_time
                return result
                
            except Exception as e:
                print(f"Ошибка получения текущей погоды: {e}")
                if self.cached_weather:
                    return self.cached_weather
                return {"temp": "—", "wind": "—"}

    async def get_city_average_history(self):
        """Стягивает PM2.5 и 7 дней истории (с кэшированием на 1 час)"""
        current_time = time.time()
        if self.cached_history and (current_time - self.history_time) < self.cache_ttl:
            return self.cached_history

        url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "current": ["pm2_5", "pm10"], 
            "hourly": ["pm2_5", "pm10"],
            "past_days": 7, 
            "forecast_days": 1,
            "timezone": "Asia/Almaty"
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, params=params, timeout=10.0)
                response.raise_for_status()
                data = response.json()
                
                current_pm25 = data["current"]["pm2_5"]
                current_pm10 = data["current"]["pm10"]
                updated_at = pd.to_datetime(data["current"]["time"]).strftime('%H:%M')
                
                df = pd.DataFrame(data['hourly'])
                df['time'] = pd.to_datetime(df['time'])
                df = df.dropna(subset=['pm2_5', 'pm10'])
                
                daily_mean = df.groupby(df['time'].dt.date)['pm2_5'].mean()
                roll7_pm25 = daily_mean.mean()
                
                last_24h = df.tail(24)
                trend_labels = last_24h['time'].dt.strftime('%H:00').tolist()
                trend_values = last_24h['pm2_5'].round(1).tolist()
                
                result = {
                    "current_pm25": float(current_pm25),
                    "current_pm10": float(current_pm10),
                    "roll7_pm25": float(roll7_pm25),
                    "trend_labels": trend_labels,
                    "trend_values": trend_values,
                    "updated_at": updated_at
                }
                
                # Сохраняем в кэш
                self.cached_history = result
                self.history_time = current_time
                return result
                
            except Exception as e:
                print(f"Критическая ошибка fetcher: {e}")
                if self.cached_history:
                    return self.cached_history
                return {
                    "current_pm25": 0, "current_pm10": 0, "roll7_pm25": 0,
                    "trend_labels": [], "trend_values": [], 
                    "updated_at": "Ошибка сети"
                }

# Создаем ОДИН глобальный объект (Singleton) при старте сервера
om_fetcher = OpenMeteoFetcher()


class OpenAQFetcher:
    def __init__(self):
        self.anchor_stations = [
            {"name": "CHP", "lat": 43.29161, "lon": 76.80437},
            {"name": "School 137", "lat": 43.31676, "lon": 76.91084},
            {"name": "Kokzhiyek-63", "lat": 43.35813, "lon": 76.9229},
            {"name": "EkoPost", "lat": 43.24966, "lon": 76.8056},
            {"name": "Alatau", "lat": 43.17658, "lon": 76.89771},
            {"name": "Elaman 105 St.", "lat": 43.36861, "lon": 76.94705},
            {"name": "Atyrau 54", "lat": 43.29678, "lon": 76.99159},
            {"name": "Kokkainar Micro", "lat": 43.29076, "lon": 76.84123},
            {"name": "KBTU", "lat": 43.25348, "lon": 76.94537},
            {"name": "Iliyski trakt", "lat": 43.34158, "lon": 76.98193},
            {"name": "School 187", "lat": 43.18579, "lon": 76.82848},
            {"name": "Mamyr-3", "lat": 43.21431, "lon": 76.8555},
            {"name": "Gymnasium 39", "lat": 43.34819, "lon": 76.8433},
            {"name": "Hospital-7", "lat": 43.23396, "lon": 76.80013},
            {"name": "Ryskulova 81", "lat": 43.27865, "lon": 76.90347},
            {"name": "School 77", "lat": 43.21846, "lon": 76.94964},
            {"name": "School 190", "lat": 43.15476, "lon": 76.8992},
            {"name": "Nicolas International", "lat": 43.19365, "lon": 76.90963},
            {"name": "School 192", "lat": 43.17203, "lon": 76.85234},
            {"name": "Respublika 4", "lat": 43.2368, "lon": 76.94481},
            {"name": "Kotelnikova St.", "lat": 43.3104, "lon": 76.94217},
            {"name": "AsiaFood", "lat": 43.24283, "lon": 76.82872},
            {"name": "Zhetysu 47", "lat": 43.29171, "lon": 76.9898},
            {"name": "DIS-7", "lat": 43.20533, "lon": 76.77692}
        ]
        self.imputer = SpatialImputer(radius_steps=[1.0, 2.0, 3.0, 5.0])
        
        # --- СИСТЕМА КЭШИРОВАНИЯ OPENAQ ---
        self.cached_24_nodes = None
        self.cached_24_sources = None
        self.last_fetch_time = 0
        self.cache_ttl = 7200  # Обновление раз в 2 часа (7200 секунд)

    def _fallback_24_nodes(self):
        if self.cached_24_nodes:
            return self.cached_24_nodes

        try:
            url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude=43.25&longitude=76.95&current=pm2_5&timezone=Asia/Almaty"
            resp = requests.get(url, timeout=5).json()
            fallback_pm25 = float(resp["current"]["pm2_5"])
        except Exception as e:
            print(f"Ошибка получения фоллбэка STGCN: {e}")
            fallback_pm25 = 40.0 

        if not math.isfinite(fallback_pm25) or fallback_pm25 <= 0.5:
            fallback_pm25 = 40.0
            
        self.cached_24_sources = ["imputed"] * len(self.anchor_stations)
        return [fallback_pm25] * len(self.anchor_stations)

    def get_realtime_24_nodes(self):
        current_time = time.time()
        
        # 1. Возврат кэша (модель не дергает API лишний раз)
        if self.cached_24_nodes and (current_time - self.last_fetch_time) < self.cache_ttl:
            print("Отдаю данные OpenAQ из памяти (API не вызывался).")
            return self.cached_24_nodes

        print("Кэш устарел. Начинаю безопасную выгрузку OpenAQ v3 (без риска бана)...")
        API_KEY = "YOUR_OPENAQ_API_KEY"
        headers = {"accept": "application/json", "X-API-Key": API_KEY}
        
        sensor_tasks = []
        live_stations_data = [] 
        
        try:
            loc_url = "https://api.openaq.org/v3/locations"
            loc_params = {"bbox": "76.7,43.1,77.2,43.5", "limit": 200}
            
            loc_resp = requests.get(loc_url, params=loc_params, headers=headers, timeout=10)
            loc_resp.raise_for_status()
            locations = loc_resp.json().get('results', [])
            
            now = datetime.datetime.utcnow()
            
            for loc in locations:
                dt_last_str = loc.get("datetimeLast", {}).get("utc")
                if not dt_last_str: continue
                try:
                    dt = datetime.datetime.fromisoformat(dt_last_str.replace('Z', '+00:00')).replace(tzinfo=None)
                    if (now - dt).total_seconds() > 7 * 24 * 3600:
                        continue 
                except:
                    pass
                
                for sensor in loc.get("sensors", []):
                    param = sensor.get("parameter", {})
                    if param.get("id") == 2 or param.get("name", "").lower() in ["pm25", "pm2.5"]:
                        s_id = sensor.get("id")
                        if s_id:
                            sensor_tasks.append({
                                'sensor_id': s_id,
                                'loc_name': loc.get("name", "Unknown")
                            })
                        break

            # ШАГ 2: БЕЗОПАСНАЯ ПОСЛЕДОВАТЕЛЬНАЯ ВЫГРУЗКА
            # Берем только топ-20 сенсоров, чтобы не превысить лимит (60 запросов в минуту)
            sensor_tasks = sensor_tasks[:20] 
            print(f"[API] Запрашиваю {len(sensor_tasks)} сенсоров PM2.5...")
            
            for task in sensor_tasks:
                s_url = f"https://api.openaq.org/v3/sensors/{task['sensor_id']}"
                try:
                    s_resp = requests.get(s_url, headers=headers, timeout=5)
                    if s_resp.status_code == 200:
                        s_data = s_resp.json().get('results', [])
                        if s_data:
                            latest = s_data[0].get("latest", {})
                            val = latest.get("value")
                            coords = latest.get("coordinates", {})
                            lat = coords.get("latitude")
                            lon = coords.get("longitude")
                            
                            if lat and lon and val is not None and 0.5 < float(val) < 2000.0:
                                live_stations_data.append({
                                    'name': task['loc_name'],
                                    'lat': float(lat),
                                    'lon': float(lon),
                                    'pm25': float(val)
                                })
                except:
                    pass
                # Задержка полсекунды между запросами гарантирует отсутствие бана
                time.sleep(0.5) 
                        
        except Exception as e:
            print(f"Ошибка API OpenAQ v3: {e}")

        # ФОЛЛБЭК
        if not live_stations_data:
            print("Живых сенсоров не найдено. Применяю фоллбэк к Open-Meteo...")
            if self.cached_24_nodes:
                return self.cached_24_nodes
            return self._fallback_24_nodes()

        live_df = pd.DataFrame(live_stations_data)
        live_df = live_df.groupby(['lat', 'lon'], as_index=False).agg({
            'name': 'first',
            'pm25': 'mean'
        })
        
        if live_df.empty:
            return self._fallback_24_nodes()

        # ШАГ 3: Применяем алгоритм Spatial Imputer
        city_mean_pm25 = float(live_df['pm25'].astype(float).mean())
        final_24_values = []
        final_24_sources = []
        
        for anchor in self.anchor_stations:
            exact_match = live_df[
                (abs(live_df['lat'] - anchor['lat']) < 0.001) & 
                (abs(live_df['lon'] - anchor['lon']) < 0.001)
            ]
            
            if not exact_match.empty:
                val = float(exact_match.iloc[0]['pm25'])
                source = "real"
            else:
                val = self.imputer.impute(anchor['name'], anchor, live_df)
                source = "imputed"
                
            if not math.isfinite(float(val)) or float(val) <= 0.5:
                val = city_mean_pm25
                source = "imputed"

            final_24_values.append(val)
            final_24_sources.append(source)

        # 4. Обновляем кэш
        self.cached_24_nodes = final_24_values
        self.cached_24_sources = final_24_sources
        self.last_fetch_time = current_time
        
        return final_24_values
    
# Создаем ОДИН глобальный объект (Singleton) при старте сервера
openaq_fetcher = OpenAQFetcher()