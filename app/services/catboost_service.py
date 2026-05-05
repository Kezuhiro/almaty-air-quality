import os
import datetime
import time  # Добавлено для кэширования
import numpy as np
import httpx  # Обязательно pip install httpx для асинхронности
from catboost import CatBoostRegressor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models", "catboost_daily", "artifacts", "catboost_final.cbm")

class CatBoostRunner:
    def __init__(self):
        self.model = CatBoostRegressor()
        try:
            self.model.load_model(MODEL_PATH)
            print(f"CatBoost успешно загружен из: {MODEL_PATH}")
        except Exception as e:
            print(f"Критическая ошибка загрузки модели CatBoost: {e}")
            self.model = None
            
        # --- СИСТЕМА КЭШИРОВАНИЯ ---
        self.cached_weather = None
        self.weather_time = 0
        self.cache_ttl = 3600  # Время жизни кэша: 1 час (3600 секунд)

    async def _fetch_tomorrow_weather(self):
        """Асинхронно получаем прогноз погоды ИМЕННО НА ЗАВТРА (с кэшированием)"""
        current_time = time.time()
        
        # Возвращаем кэш, если данные были загружены менее часа назад
        if self.cached_weather and (current_time - self.weather_time) < self.cache_ttl:
            return self.cached_weather

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": 43.25,
            "longitude": 76.95,
            "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", 
                      "wind_speed_10m_max", "wind_direction_10m_dominant"],
            "timezone": "Asia/Almaty",
            "forecast_days": 2 # 0 - сегодня, 1 - завтра
        }
        
        # Используем асинхронный клиент
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, params=params, timeout=10.0)
                response.raise_for_status()
                data = response.json()
                
                daily = data['daily']
                result = {
                    'temp_max': daily['temperature_2m_max'][1],
                    'temp_min': daily['temperature_2m_min'][1],
                    'temp_mean': (daily['temperature_2m_max'][1] + daily['temperature_2m_min'][1]) / 2,
                    'precip': daily['precipitation_sum'][1],
                    'wind_speed_max': daily['wind_speed_10m_max'][1] / 3.6, # км/ч в м/с
                    'wind_speed_mean': (daily['wind_speed_10m_max'][1] / 3.6) * 0.7,
                    'wind_dir': daily['wind_direction_10m_dominant'][1],
                    'pressure': 920.0 # В идеале тоже парсить часовой прогноз на завтра
                }
                
                # Сохраняем успешный ответ в кэш
                self.cached_weather = result
                self.weather_time = current_time
                return result
                
            except Exception as e:
                # Никаких дефолтных значений. Если API упал, нужно бросать ошибку, 
                # чтобы сервер выдал нормальный HTTP 500 Service Unavailable, а не врал юзеру.
                print(f"Ошибка получения прогноза погоды на завтра: {e}")
    
                if self.cached_weather:
                    return self.cached_weather
                raise e

    async def predict_tomorrow(self, current_pm25: float, current_pm10: float, roll7_pm25: float):
        """
        current_pm25, current_pm10, roll7_pm25 - это лаги (вчерашние/сегодняшние данные),
        которые ты можешь вытаскивать из DataManager.get_last_features() прямо в роутере FastAPI.
        """
        if self.model is None:
            raise RuntimeError("Модель CatBoost не инициализирована.")

        # 1. Ждем ответа от API погоды (теперь с кэшем)
        w = await self._fetch_tomorrow_weather()
        
        # 2. Считаем Feature Engineering на основе ЗАВТРАШНЕЙ погоды
        heating_intensity = max(0, 18 - w['temp_mean'])
        is_stagnation = 1 if (w['wind_speed_mean'] < 2.0 and w['precip'] < 0.1) else 0
        ventilation = w['wind_speed_mean'] * w['temp_max']
        inversion_potential = w['pressure'] * (1 / (w['wind_speed_mean'] + 0.1))
        wind_from_mountains = 1 if (135 <= w['wind_dir'] <= 225) else 0
        
        tomorrow = datetime.date.today() + datetime.timedelta(days=1)
        month_sin = np.sin(2 * np.pi * tomorrow.month / 12)
        month_cos = np.cos(2 * np.pi * tomorrow.month / 12)

        # 3. Строгий порядок 18 фичей для инференса
        features = [
            current_pm25,         
            current_pm10,         
            roll7_pm25,           
            w['temp_mean'],       
            w['temp_min'],        
            w['temp_max'],        
            w['wind_speed_mean'], 
            w['wind_speed_max'],  
            w['precip'],          
            w['pressure'],        
            heating_intensity,    
            is_stagnation,        
            ventilation,          
            inversion_potential,  
            wind_from_mountains,  
            month_sin,            
            month_cos,            
            tomorrow.weekday()    
        ]

        # 4. Прогноз
        pred = self.model.predict([features])[0]
        pred = max(0, pred) 
        
        if pred <= 35:
            advice = "Воздух чистый. Отличное время для прогулок и проветривания!"
        elif pred <= 75:
            advice = "Качество воздуха приемлемое. Чувствительным людям стоит ограничить активность на улице."
        else:
            advice = "Ожидается смог! Рекомендуется закрыть окна и использовать очистители воздуха."
            
        return {
            "date": tomorrow.strftime("%d.%m.%Y"),
            "pm25": round(pred, 1),
            "aqi": int(pred * 2.5), 
            "temp": round(w['temp_mean']),
            "wind": round(w['wind_speed_mean'], 1),
            "pressure": round(w['pressure']),
            "humidity": 44, # Заглушка, лучше брать из API
            "precip": round(w['precip'], 1),
            "advice": advice
        }

# В main.py ты должен импортировать этот объект
catboost_runner = CatBoostRunner()