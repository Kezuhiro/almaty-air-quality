import math
import os
import sys
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
sys.path.append(BASE_DIR)

from app.db.database import get_sync_database_url, get_sync_engine, replace_table
from app.db.models import Base, DailyFeature, HourlyFeature, Measurement, Station, StationDistance

CITYWIDE_STATION_ID = 999

def haversine(lat1, lon1, lat2, lon2):
    radius = 6371.0
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(delta_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius * c

def _seed_stations(engine) -> None:
    with Session(engine) as session:
        if session.query(Station).first() is not None:
            print("Таблица stations уже заполнена, пропускаю.")
            return
        
        stations_path = os.path.join(DATA_DIR, "almaty_stations_coordinates.csv")
        if not os.path.exists(stations_path):
            return

        df_stations = pd.read_csv(stations_path)
        for _, row in df_stations.iterrows():
            session.add(Station(
                id=int(row["location_id"]),
                name=str(row["name"]),
                lat=float(row["latitude"]),
                lon=float(row["longitude"])
            ))
        session.add(Station(id=CITYWIDE_STATION_ID, name="Almaty_Citywide", lat=43.25, lon=76.95))
        session.commit()
        print(f"В PostgreSQL добавлено станций: {len(df_stations) + 1}")

def _seed_distances(engine) -> None:
    with Session(engine) as session:
        if session.query(StationDistance).first() is not None:
            print("Таблица station_distances уже заполнена, пропускаю.")
            return

        stations = session.query(Station).filter(Station.id != CITYWIDE_STATION_ID).all()
        count = 0
        for s1 in stations:
            for s2 in stations:
                if s1.id != s2.id:
                    dist = haversine(s1.lat, s1.lon, s2.lat, s2.lon)
                    session.add(StationDistance(source_id=s1.id, target_id=s2.id, distance=dist))
                    count += 1
        session.commit()
        print(f"В PostgreSQL добавлено связей графа: {count}")

def _seed_measurements(engine) -> None:
    with Session(engine) as session:
        if session.query(Measurement).first() is not None:
            print("Таблица measurements уже заполнена, пропускаю.")
            return

    # Загрузка точечных замеров
    matrix_path = os.path.join(DATA_DIR, "almaty_pm25_matrix.csv")
    if os.path.exists(matrix_path):
        df_pm25 = pd.read_csv(matrix_path)
        df_pm25["datetime_utc"] = pd.to_datetime(df_pm25["datetime_utc"])
        df_melted = df_pm25.melt(id_vars=["datetime_utc"], var_name="station_id", value_name="pm25")
        df_insert = pd.DataFrame({
            "station_id": df_melted["station_id"].astype(int),
            "timestamp": df_melted["datetime_utc"],
            "pm25": df_melted["pm25"],
            "pm10": None,
            "is_imputed": False
        })
        df_insert.to_sql("measurements", engine, if_exists="append", index=False, chunksize=5000)
        print(f"В PostgreSQL добавлено точечных замеров: {len(df_insert)}")

    # Загрузка общегородских замеров
    hourly_path = os.path.join(DATA_DIR, "train_hourly_complete.csv")
    if os.path.exists(hourly_path):
        df_hourly = pd.read_csv(hourly_path)
        df_hourly["datetime"] = pd.to_datetime(df_hourly["datetime"])
        
        df_hourly["datetime"] = df_hourly["datetime"].dt.tz_localize('Asia/Almaty', ambiguous='NaT', nonexistent='shift_forward')
        df_hourly = df_hourly.dropna(subset=["datetime"])

        df_insert_city = pd.DataFrame({
            "station_id": CITYWIDE_STATION_ID,
            "timestamp": df_hourly["datetime"],
            "pm25": df_hourly.get("pm25", None),
            "pm10": df_hourly.get("pm10", None),
            "is_imputed": False
        }).drop_duplicates(subset=["timestamp"])

        df_insert_city.to_sql("measurements", engine, if_exists="append", index=False, chunksize=5000)
        print(f"В PostgreSQL добавлено общегородских замеров: {len(df_insert_city)}")

def _seed_daily_features(engine) -> None:
    daily_path = os.path.join(DATA_DIR, "train_dataset_full.csv")
    if os.path.exists(daily_path):
        df_daily = pd.read_csv(daily_path)
        df_daily["date"] = pd.to_datetime(df_daily["date"]).dt.date
        replace_table("daily_features", df_daily)
        print(f"В PostgreSQL загружено суточных строк: {len(df_daily)}")

def _seed_hourly_features(engine) -> None:
    hourly_path = os.path.join(DATA_DIR, "train_hourly_complete.csv")
    if os.path.exists(hourly_path):
        df_hourly = pd.read_csv(hourly_path)
        df_hourly["datetime"] = pd.to_datetime(df_hourly["datetime"])
        if df_hourly["datetime"].dt.tz is not None:
            df_hourly["datetime"] = df_hourly["datetime"].dt.tz_localize(None)
        replace_table("hourly_features", df_hourly)
        print(f"В PostgreSQL загружено часовых строк: {len(df_hourly)}")

def seed_database():
    print(f"Подключение к БД: {get_sync_database_url()}")
    engine = get_sync_engine()
    
    # Создаем таблицы в БД перед заполнением
    Base.metadata.create_all(engine)

    _seed_stations(engine)
    _seed_distances(engine)
    _seed_measurements(engine)
    _seed_daily_features(engine)
    _seed_hourly_features(engine)
    print("🚀 Все данные успешно синхронизированы!")

if __name__ == "__main__":
    seed_database()