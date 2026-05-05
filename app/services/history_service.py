import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")

class HistoryService:
    def __init__(self):
        self.daily_data = None
        self.hourly_data = None
        self.station_data = None
        self.load_data()

    def load_data(self):
        try:
            # 1. Daily
            daily_path = os.path.join(DATA_DIR, "train_dataset_full.csv")
            if os.path.exists(daily_path):
                df_daily = pd.read_csv(daily_path)
                df_daily['date'] = pd.to_datetime(df_daily['date'])
                df_daily = df_daily.sort_values('date')
                self.daily_data = df_daily

            # 2. Hourly
            hourly_path = os.path.join(DATA_DIR, "train_hourly_complete.csv")
            if os.path.exists(hourly_path):
                df_hourly = pd.read_csv(hourly_path)
                df_hourly['datetime'] = pd.to_datetime(df_hourly['datetime'])
                # Let's take only last 30 days for hourly to save memory/bandwidth
                last_date = df_hourly['datetime'].max()
                df_hourly = df_hourly[df_hourly['datetime'] >= (last_date - pd.Timedelta(days=30))]
                df_hourly = df_hourly.sort_values('datetime')
                self.hourly_data = df_hourly

            # 3. Stations
            stations_path = os.path.join(DATA_DIR, "almaty_pm25_matrix.csv")
            if os.path.exists(stations_path):
                df_stations = pd.read_csv(stations_path)
                df_stations['datetime_utc'] = pd.to_datetime(df_stations['datetime_utc'])
                
                # Exclude specified stations
                exclude = ['2812676', '2812831', '2812649', '2812716', '2812691']
                cols_to_keep = [c for c in df_stations.columns if c not in exclude]
                df_stations = df_stations[cols_to_keep]
                
                self.station_data = df_stations

        except Exception as e:
            print(f"Error loading history data: {e}")
    
    def get_context(self):

        context = {}
        
        # Daily
        if self.daily_data is not None:
            df = self.daily_data
            df = df.reset_index(drop=True) 
            
            context["stats"] = {
                "days": len(df),
                "avg": round(df['pm25'].mean(), 1),
                "max": round(df['pm25'].max(), 1)
            }
            context["chart_dates"] = df['date'].dt.strftime('%Y-%m-%d').tolist()
            context["chart_values"] = df['pm25'].round(1).tolist()
            
            history_list = df.tail(365).sort_values('date', ascending=False).to_dict('records')
            context["history"] = [
                {
                    "date": row['date'].strftime('%Y-%m-%d'),
                    "pm25": round(row['pm25'], 1),
                    "pm10": round(row['pm10'], 1) if not pd.isna(row.get('pm10')) else 0
                }
                for row in history_list
            ]
        
        # Hourly
        if self.hourly_data is not None:
            df_h = self.hourly_data.tail(24 * 7) # Last 7 days
            context["hourly_dates"] = df_h['datetime'].dt.strftime('%d.%m %H:00').tolist()
            context["hourly_values"] = df_h['pm25'].round(1).tolist()
            
        # Stations
        if self.station_data is not None:
            df_s = self.station_data
            numeric_cols = [c for c in df_s.columns if c != 'datetime_utc']
            station_avgs = df_s[numeric_cols].mean().round(1).to_dict()
            
            sorted_stations = sorted([{"id": k, "avg_pm25": v} for k, v in station_avgs.items()], key=lambda x: x['avg_pm25'], reverse=True)
            context["stations"] = sorted_stations
            
            # Station history (last 7 days average across all included stations)
            df_s_recent = df_s.tail(24 * 7).copy()
            df_s_recent['avg_all'] = df_s_recent[numeric_cols].mean(axis=1)
            context["station_dates"] = df_s_recent['datetime_utc'].dt.strftime('%d.%m %H:00').tolist()
            context["station_values"] = df_s_recent['avg_all'].round(1).tolist()

        if not context:
            context["error"] = "Данные не найдены"
            
        return context

history_service = HistoryService()
