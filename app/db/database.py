import os
from functools import lru_cache
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import insert
from .models import Base

DEFAULT_DB_URL = os.getenv(
    "SYNC_DATABASE_URL"
)

@lru_cache(maxsize=1)
def get_sync_database_url() -> str:
    url = os.getenv("SYNC_DATABASE_URL")
    return url if url else DEFAULT_DB_URL

@lru_cache(maxsize=1)
def get_sync_engine():
    engine = create_engine(get_sync_database_url(), pool_pre_ping=True)
    Base.metadata.create_all(engine)
    return engine

def read_sql_query(query: str, params: dict | None = None, parse_dates: list[str] | None = None) -> pd.DataFrame:
    return pd.read_sql_query(text(query), get_sync_engine(), params=params, parse_dates=parse_dates)

def insert_on_conflict_nothing(table, conn, keys, data_iter):
    data = [dict(zip(keys, row)) for row in data_iter]
    stmt = insert(table.table).values(data).on_conflict_do_nothing()
    conn.execute(stmt)

def append_data(table_name: str, dataframe: pd.DataFrame) -> None:
    engine = get_sync_engine()
    dataframe.to_sql(
        table_name, 
        engine, 
        if_exists="append", 
        index=False, 
        method=insert_on_conflict_nothing, 
        chunksize=5000
    )

def replace_table(table_name: str, dataframe: pd.DataFrame) -> None:
    """Быстрый метод перезаписи без блокировки транзакций (удален метод multi)"""
    engine = get_sync_engine()
    with engine.begin() as connection:
        connection.execute(text(f"TRUNCATE TABLE {table_name} RESTART IDENTITY"))
    dataframe.to_sql(table_name, engine, if_exists="append", index=False, chunksize=5000)