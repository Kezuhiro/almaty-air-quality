import pandas as pd

from db.database import read_sql_query


CITYWIDE_STATION_ID = 999
EXCLUDED_STATION_IDS = [2812676, 2812831, 2812649, 2812716, 2812691]
EXCLUDED_STATION_IDS_SQL = ", ".join(str(station_id) for station_id in EXCLUDED_STATION_IDS)


class HistoryService:
    def _load_daily_data(self) -> pd.DataFrame:
        query = """
            SELECT "date", pm25, pm10
            FROM daily_features
            ORDER BY "date"
        """
        return read_sql_query(query, parse_dates=["date"])

    def _load_hourly_data(self) -> pd.DataFrame:
        query = """
            SELECT "datetime", pm25
            FROM hourly_features
            WHERE "datetime" >= (
                SELECT MAX("datetime") - INTERVAL '30 days'
                FROM hourly_features
            )
            ORDER BY "datetime"
        """
        return read_sql_query(query, parse_dates=["datetime"])

    def _load_station_averages(self) -> pd.DataFrame:
        query = f"""
            SELECT station_id, AVG(pm25) AS avg_pm25
            FROM measurements
            WHERE station_id <> {CITYWIDE_STATION_ID}
              AND station_id NOT IN ({EXCLUDED_STATION_IDS_SQL})
              AND pm25 IS NOT NULL
            GROUP BY station_id
            ORDER BY avg_pm25 DESC
        """
        return read_sql_query(query)

    def _load_station_recent_series(self) -> pd.DataFrame:
        query = f"""
            WITH latest AS (
                SELECT MAX(timestamp) AS max_ts
                FROM measurements
                WHERE station_id <> {CITYWIDE_STATION_ID}
                  AND station_id NOT IN ({EXCLUDED_STATION_IDS_SQL})
                  AND pm25 IS NOT NULL
            )
            SELECT timestamp AS datetime_utc, AVG(pm25) AS avg_all
            FROM measurements, latest
            WHERE station_id <> {CITYWIDE_STATION_ID}
              AND station_id NOT IN ({EXCLUDED_STATION_IDS_SQL})
              AND pm25 IS NOT NULL
              AND timestamp >= latest.max_ts - INTERVAL '7 days'
            GROUP BY timestamp
            ORDER BY timestamp
        """
        return read_sql_query(query, parse_dates=["datetime_utc"])

    def get_context(self) -> dict:
        context: dict = {}

        try:
            daily_data = self._load_daily_data()
            if not daily_data.empty:
                daily_data = daily_data.sort_values("date").reset_index(drop=True)
                context["stats"] = {
                    "days": len(daily_data),
                    "avg": round(daily_data["pm25"].mean(), 1),
                    "max": round(daily_data["pm25"].max(), 1),
                }
                context["chart_dates"] = daily_data["date"].dt.strftime("%Y-%m-%d").tolist()
                context["chart_values"] = daily_data["pm25"].round(1).tolist()

                history_list = daily_data.tail(365).sort_values("date", ascending=False).to_dict("records")
                context["history"] = [
                    {
                        "date": row["date"].strftime("%Y-%m-%d"),
                        "pm25": round(row["pm25"], 1),
                        "pm10": round(row["pm10"], 1) if not pd.isna(row.get("pm10")) else 0,
                    }
                    for row in history_list
                ]

            hourly_data = self._load_hourly_data()
            if not hourly_data.empty:
                hourly_slice = hourly_data.tail(24 * 7)
                context["hourly_dates"] = hourly_slice["datetime"].dt.strftime("%d.%m %H:00").tolist()
                context["hourly_values"] = hourly_slice["pm25"].round(1).tolist()

            station_averages = self._load_station_averages()
            if not station_averages.empty:
                context["stations"] = [
                    {"id": str(row["station_id"]), "avg_pm25": round(row["avg_pm25"], 1)}
                    for _, row in station_averages.iterrows()
                ]

            station_recent = self._load_station_recent_series()
            if not station_recent.empty:
                context["station_dates"] = station_recent["datetime_utc"].dt.strftime("%d.%m %H:00").tolist()
                context["station_values"] = station_recent["avg_all"].round(1).tolist()

            if not context:
                context["error"] = "Данные не найдены в PostgreSQL"
        except Exception as exc:
            print(f"Error loading history data from PostgreSQL: {exc}")
            context["error"] = "Не удалось загрузить историю из PostgreSQL"

        return context


history_service = HistoryService()
