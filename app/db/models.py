from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Station(Base):
    __tablename__ = "stations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    lat: Mapped[float] = mapped_column(Float, nullable=False)
    lon: Mapped[float] = mapped_column(Float, nullable=False)

    measurements = relationship("Measurement", back_populates="station")


class StationDistance(Base):
    __tablename__ = "station_distances"

    source_id: Mapped[int] = mapped_column(ForeignKey("stations.id"), primary_key=True)
    target_id: Mapped[int] = mapped_column(ForeignKey("stations.id"), primary_key=True)
    distance: Mapped[float] = mapped_column(Float, nullable=False)


class Measurement(Base):
    __tablename__ = "measurements"
    __table_args__ = (
        UniqueConstraint("station_id", "timestamp", name="uix_station_timestamp"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    station_id: Mapped[int] = mapped_column(ForeignKey("stations.id"), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    pm25: Mapped[float | None] = mapped_column(Float, nullable=True)
    pm10: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_imputed: Mapped[bool] = mapped_column(Boolean, default=False)

    station = relationship("Station", back_populates="measurements")


class Weather(Base):
    __tablename__ = "weather"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, unique=True, index=True)
    wind_speed: Mapped[float | None] = mapped_column(Float, nullable=True)
    wind_dir: Mapped[float | None] = mapped_column(Float, nullable=True)
    temp: Mapped[float | None] = mapped_column(Float, nullable=True)
    humidity: Mapped[float | None] = mapped_column(Float, nullable=True)
    pressure: Mapped[float | None] = mapped_column(Float, nullable=True)


class Forecast(Base):
    __tablename__ = "forecasts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    target_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    model_name: Mapped[str] = mapped_column(String, nullable=False)
    station_id: Mapped[int | None] = mapped_column(ForeignKey("stations.id"), nullable=True)
    pm25_pred: Mapped[float] = mapped_column(Float, nullable=False)


class DailyFeature(Base):
    __tablename__ = "daily_features"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[date] = mapped_column(Date, nullable=False, unique=True, index=True)

    pm25: Mapped[float | None] = mapped_column(Float, nullable=True)
    pm10: Mapped[float | None] = mapped_column(Float, nullable=True)
    so2: Mapped[float | None] = mapped_column(Float, nullable=True)
    co: Mapped[float | None] = mapped_column(Float, nullable=True)
    temp_mean: Mapped[float | None] = mapped_column(Float, nullable=True)
    temp_min: Mapped[float | None] = mapped_column(Float, nullable=True)
    temp_max: Mapped[float | None] = mapped_column(Float, nullable=True)
    precip: Mapped[float | None] = mapped_column(Float, nullable=True)
    wind_speed_max: Mapped[float | None] = mapped_column(Float, nullable=True)
    wind_speed_mean: Mapped[float | None] = mapped_column(Float, nullable=True)
    wind_dir: Mapped[float | None] = mapped_column(Float, nullable=True)
    pressure: Mapped[float | None] = mapped_column(Float, nullable=True)
    month: Mapped[int | None] = mapped_column(Integer, nullable=True)
    day_of_week: Mapped[int | None] = mapped_column(Integer, nullable=True)
    month_sin: Mapped[float | None] = mapped_column(Float, nullable=True)
    month_cos: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_heating_season: Mapped[int | None] = mapped_column(Integer, nullable=True)
    pm25_lag1: Mapped[float | None] = mapped_column(Float, nullable=True)
    pm25_lag3: Mapped[float | None] = mapped_column(Float, nullable=True)
    pm25_lag7: Mapped[float | None] = mapped_column(Float, nullable=True)
    pm10_lag1: Mapped[float | None] = mapped_column(Float, nullable=True)
    pm25_roll7_mean: Mapped[float | None] = mapped_column(Float, nullable=True)
    pm25_roll7_std: Mapped[float | None] = mapped_column(Float, nullable=True)
    wind_dir_sin: Mapped[float | None] = mapped_column(Float, nullable=True)
    wind_dir_cos: Mapped[float | None] = mapped_column(Float, nullable=True)
    wind_from_mountains: Mapped[int | None] = mapped_column(Integer, nullable=True)
    heating_intensity: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_stagnation: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ventilation: Mapped[float | None] = mapped_column(Float, nullable=True)
    inversion_potential: Mapped[float | None] = mapped_column(Float, nullable=True)


class HourlyFeature(Base):
    __tablename__ = "hourly_features"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    datetime: Mapped[datetime] = mapped_column(DateTime, nullable=False, unique=True, index=True)

    temp: Mapped[float | None] = mapped_column(Float, nullable=True)
    precip: Mapped[float | None] = mapped_column(Float, nullable=True)
    wind_speed: Mapped[float | None] = mapped_column(Float, nullable=True)
    pressure: Mapped[float | None] = mapped_column(Float, nullable=True)
    pm10: Mapped[float | None] = mapped_column(Float, nullable=True)
    pm25: Mapped[float | None] = mapped_column(Float, nullable=True)
    co: Mapped[float | None] = mapped_column(Float, nullable=True)
    so2: Mapped[float | None] = mapped_column(Float, nullable=True)
    day_of_week: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_heating_season: Mapped[int | None] = mapped_column(Integer, nullable=True)
    hour_sin: Mapped[float | None] = mapped_column(Float, nullable=True)
    hour_cos: Mapped[float | None] = mapped_column(Float, nullable=True)
    month_sin: Mapped[float | None] = mapped_column(Float, nullable=True)
    month_cos: Mapped[float | None] = mapped_column(Float, nullable=True)
    wind_dir_sin: Mapped[float | None] = mapped_column(Float, nullable=True)
    wind_dir_cos: Mapped[float | None] = mapped_column(Float, nullable=True)
    pm25_lag1: Mapped[float | None] = mapped_column(Float, nullable=True)
    pm25_lag2: Mapped[float | None] = mapped_column(Float, nullable=True)
    pm25_lag3: Mapped[float | None] = mapped_column(Float, nullable=True)
    pm25_lag24: Mapped[float | None] = mapped_column(Float, nullable=True)
    pm25_roll24_mean: Mapped[float | None] = mapped_column(Float, nullable=True)
    pm25_roll24_std: Mapped[float | None] = mapped_column(Float, nullable=True)
    pm25_lag48: Mapped[float | None] = mapped_column(Float, nullable=True)
    pm25_lag72: Mapped[float | None] = mapped_column(Float, nullable=True)
    pm25_wind_interaction: Mapped[float | None] = mapped_column(Float, nullable=True)