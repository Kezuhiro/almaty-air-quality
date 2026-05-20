import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle
import folium
import requests
from folium.plugins import HeatMap
from services.fetcher import openaq_fetcher


# 1. АРХИТЕКТУРА МОДЕЛИ

class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        support = torch.matmul(x, self.weight) 
        out = torch.einsum('vw,bswf->bsvf', adj, support)
        return F.relu(out)

class AlmatySTGCN(nn.Module):
    def __init__(self, num_nodes, num_features, hidden_dim, seq_len):
        super().__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.gcn = GraphConvLayer(in_features=num_features, out_features=hidden_dim)
        self.gru = nn.GRU(input_size=num_nodes * hidden_dim, hidden_size=hidden_dim * 2, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, num_nodes)

    def forward(self, x, adj):
        batch_size = x.size(0)
        x_gcn = self.gcn(x, adj)
        x_gru_in = x_gcn.reshape(batch_size, self.seq_len, -1)
        gru_out, _ = self.gru(x_gru_in)
        last_step = gru_out[:, -1, :] 
        out = self.fc(last_step)
        return out

# 2. КЛАСС ДЛЯ ИНФЕРЕНСА (STGCN Runner)

class STGCNRunner:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.adj_tensor = None
        self.scaler = None
        self.stations = openaq_fetcher.anchor_stations
        
        # --- СИСТЕМА КЭШИРОВАНИЯ ---
        self.cached_history = None
        self.history_time = 0
        
        self.cached_predictions = None
        self.predictions_time = 0
        
        self.load_artifacts()

    def load_artifacts(self):
        try:
            adj_path = os.path.join(self.base_dir, "models", "stgcn_24h", "artifacts", "adj_matrix.npy")
            adj_matrix = np.load(adj_path)
            self.adj_tensor = torch.FloatTensor(adj_matrix).to(self.device)

            self.scaler_path = os.path.join(self.base_dir, "models", "stgcn_24h", "artifacts", "pm25_scaler.pkl")
            with open(self.scaler_path, "rb") as f:
                self.scaler = pickle.load(f)

            self.model = AlmatySTGCN(num_nodes=24, num_features=6, hidden_dim=16, seq_len=12).to(self.device)
            model_path = os.path.join(self.base_dir, "models", "stgcn_24h", "artifacts", "stgcn_almaty.pth")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            self.model.eval()
            print("STGCN: Все артефакты успешно загружены!")
        except Exception as e:
            print(f"Ошибка загрузки STGCN: {e}")

    def get_real_12h_history(self):
        """Скачивает НАСТОЯЩУЮ историю за последние 12 часов прямо сейчас"""
        current_time = time.time()
        # Возвращаем кэш, если ему меньше 1 часа (3600 секунд)
        if self.cached_history and (current_time - self.history_time) < 3600:
            return self.cached_history

        try:
            # Погода за 12 часов
            w_url = "https://api.open-meteo.com/v1/forecast"
            w_params = {
                "latitude": 43.25, "longitude": 76.95,
                "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m"],
                "past_hours": 12,
                "forecast_days": 1,
                "timezone": "Asia/Almaty"
            }
            w_resp = requests.get(w_url, params=w_params, timeout=10).json()
            
            a_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
            a_params = {
                "latitude": 43.25, "longitude": 76.95,
                "hourly": ["pm10", "pm2_5"],
                "past_hours": 12,
                "forecast_days": 1,
                "timezone": "Asia/Almaty"
            }
            a_resp = requests.get(a_url, params=a_params, timeout=10).json()
            
            result = {
                "temp": w_resp["hourly"]["temperature_2m"][:12],
                "hum":  w_resp["hourly"]["relative_humidity_2m"][:12],
                "wind": w_resp["hourly"]["wind_speed_10m"][:12],
                "dir":  w_resp["hourly"]["wind_direction_10m"][:12],
                "pm10": a_resp["hourly"]["pm10"][:12],
                "pm25": a_resp["hourly"]["pm2_5"][:12]
            }
            

            self.cached_history = result
            self.history_time = current_time
            return result
            
        except Exception as e:
            print(f"Ошибка выгрузки реальной истории 12ч: {e}")
            if self.cached_history:
                return self.cached_history
            return None

    def get_predictions(self):
        current_time = time.time()
        if self.cached_predictions and (current_time - self.predictions_time) < 900:
            return self.cached_predictions

        if self.model is None or self.adj_tensor is None:
            return openaq_fetcher.get_realtime_24_nodes()
            
        current_pm25 = openaq_fetcher.get_realtime_24_nodes()
        current_pm25_np = np.array(current_pm25, dtype=float)
        
        valid_current = np.isfinite(current_pm25_np) & (current_pm25_np > 0.5)
        if not valid_current.any():
            current_pm25_np = np.full(24, 40.0, dtype=float)
        else:
            current_pm25_np = np.where(valid_current, current_pm25_np, float(np.nanmean(current_pm25_np[valid_current])))

        history = self.get_real_12h_history()
        
        def fill_nans(arr, default):
            a = np.array(arr, dtype=float)
            a[np.isnan(a)] = default
            return a
            
        if history:
            h_temp = fill_nans(history["temp"], 10.0)
            h_hum  = fill_nans(history["hum"], 45.0)
            h_wind = fill_nans(history["wind"], 2.0)
            h_dir  = fill_nans(history["dir"], 180.0)
            h_pm10 = fill_nans(history["pm10"], 20.0)
            h_pm25 = fill_nans(history["pm25"], 15.0)
        else:
            h_temp, h_hum, h_wind, h_dir = np.full(12, 10.0), np.full(12, 45.0), np.full(12, 2.0), np.full(12, 180.0)
            h_pm10, h_pm25 = np.full(12, 20.0), np.full(12, 15.0)

        tensor_data = np.zeros((1, 12, 24, 6), dtype=np.float32)
        
        for i in range(24):
            station_current = current_pm25_np[i]
            city_current = h_pm25[-1] if h_pm25[-1] > 0 else 1.0
            ratio = station_current / city_current 
            
            for t in range(12):
                real_pm25_t = h_pm25[t] * ratio
                scaled_pm25 = self.scaler.transform(np.array([[real_pm25_t]]))[0][0]
                

                norm_temp = max(0.0, min(1.0, (h_temp[t] + 20) / 60.0))
                norm_hum  = max(0.0, min(1.0, h_hum[t] / 100.0))
                norm_wind = max(0.0, min(1.0, h_wind[t] / 15.0))
                norm_dir  = max(0.0, min(1.0, h_dir[t] / 360.0))
                
                tensor_data[0, t, i, 0] = scaled_pm25
                tensor_data[0, t, i, 1] = min(1.0, scaled_pm25 * 1.1) 
                tensor_data[0, t, i, 2] = norm_temp
                tensor_data[0, t, i, 3] = norm_hum
                tensor_data[0, t, i, 4] = norm_wind
                tensor_data[0, t, i, 5] = norm_dir
        
        x_tensor = torch.FloatTensor(tensor_data).to(self.device)
        
        with torch.no_grad():
            scaled_preds = self.model(x_tensor, self.adj_tensor).numpy()[0]
            real_preds = self.scaler.inverse_transform(scaled_preds.reshape(-1, 1)).flatten()
            
            final_preds = []
            for i, p in enumerate(real_preds):

                if p < 0.5 or p > 500:
                    trend_3h = h_pm25[-1] - h_pm25[-4]
                    ratio = current_pm25_np[i] / (h_pm25[-1] if h_pm25[-1] > 0 else 1.0)
                    fallback_val = current_pm25_np[i] + (trend_3h * ratio)
                    final_preds.append(max(1.0, float(fallback_val)))
                else:
                    final_preds.append(float(p))
            

            self.cached_predictions = final_preds
            self.predictions_time = current_time
            
            return final_preds

    def generate_heatmap(self):
        predictions = self.get_predictions()
        m = folium.Map(location=[43.25, 76.90], zoom_start=11, tiles="CartoDB dark_matter")
        heat_data = []
        
        for i, s in enumerate(self.stations):
            pm25_val = round(predictions[i], 1)
            folium.CircleMarker(
                location=[s["lat"], s["lon"]],
                radius=4,
                color="white",
                fill=True,
                fill_color="white",
                fill_opacity=0.7,
                tooltip=f"Узел {i+1}: {s['name']} | Прогноз: {pm25_val} мкг/м³"
            ).add_to(m)
            heat_data.append([s["lat"], s["lon"], pm25_val])

        HeatMap(
            heat_data, 
            radius=25, 
            blur=18, 
            gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'yellow', 0.8: 'orange', 1.0: 'red'}
        ).add_to(m)
        
        map_path = os.path.join(self.base_dir, "app", "templates", "stgcn_map.html")
        os.makedirs(os.path.dirname(map_path), exist_ok=True)
        m.save(map_path)


    def simulate_scenario(self, temp: float, wind: float, direction: float, hum: float, hour: int):
        if self.model is None or self.adj_tensor is None:
            return []


        current_pm25 = openaq_fetcher.get_realtime_24_nodes()
        current_pm25_np = np.array(current_pm25, dtype=float)
        valid_current = np.isfinite(current_pm25_np) & (current_pm25_np > 0.5)
        if not valid_current.any():
            current_pm25_np = np.full(24, 40.0, dtype=float)
        else:
            current_pm25_np = np.where(valid_current, current_pm25_np, float(np.nanmean(current_pm25_np[valid_current])))

        norm_hum  = max(0.0, min(1.0, hum / 100.0))
        norm_wind = max(0.0, min(1.0, wind / 15.0))
        norm_dir  = max(0.0, min(1.0, direction / 360.0))

        tensor_data = np.zeros((1, 12, 24, 6), dtype=np.float32)
        scaled_pm25 = self.scaler.transform(current_pm25_np.reshape(-1, 1)).flatten()

        for i in range(24):
            current_val = scaled_pm25[i]
            
            for t in range(12):
                hist_hour = (hour - 11 + t) % 24
                
                temp_shift = 0
                if hist_hour < 7 or hist_hour > 20:
                    temp_shift = -5.0
                elif 12 <= hist_hour <= 16:
                    temp_shift = 3.0
                
                hist_temp = temp + temp_shift
                norm_temp = max(0.0, min(1.0, (hist_temp + 20) / 60.0))

                traffic_multiplier = 1.0
                if hist_hour in [8, 9, 18, 19]:
                    traffic_multiplier = 1.25
                elif hist_hour in [2, 3, 4]:
                    traffic_multiplier = 0.8
                
                start_val = current_val * 0.7 
                step_val = start_val + (current_val - start_val) * (t / 11.0)
                step_val *= traffic_multiplier

                tensor_data[0, t, i, 0] = step_val
                tensor_data[0, t, i, 1] = min(1.0, step_val * 1.1) 
                tensor_data[0, t, i, 2] = norm_temp
                tensor_data[0, t, i, 3] = norm_hum
                tensor_data[0, t, i, 4] = norm_wind
                tensor_data[0, t, i, 5] = norm_dir

        x_tensor = torch.FloatTensor(tensor_data).to(self.device)
        
        with torch.no_grad():
            scaled_preds = self.model(x_tensor, self.adj_tensor).numpy()[0]
            real_preds = self.scaler.inverse_transform(scaled_preds.reshape(-1, 1)).flatten()
            
            results = []
            for i, p in enumerate(real_preds):
                if p < 0.5 or p > 700:
                    results.append(max(1.0, float(current_pm25_np[i])))
                else:
                    results.append(float(p))
            
            return results

stgcn_runner = STGCNRunner()