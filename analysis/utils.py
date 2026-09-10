#Shared configuration and I/O helpers for the EDA scripts.

import os
import sys
from pathlib import Path

import pandas as pd

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "output"

if load_dotenv is not None:
    load_dotenv(PROJECT_DIR / ".env")

sys.stdout.reconfigure(encoding="utf-8")

TABLE_FILES = {
    "brand": "brand.csv",
    "cpu_model": "cpu_model.csv",
    "gpu_model": "gpu_model.csv",
    "laptop_model": "laptop_model.csv",
    "laptop_benchmark_result": "laptop_benchmark_result.csv",
    "user_event_tracking": "user_event_tracking.csv",
}


def quote_identifier(identifier):
    #Quote a known identifier for ClickHouse SQL.#
    return "`" + identifier.replace("`", "``") + "`"


def create_client():
    #Create a ClickHouse client without printing credentials.#
    try:
        import clickhouse_connect
    except ImportError as error:
        raise RuntimeError(
            "Missing dependency: install dependencies with "
            "'pip install -r requirements.txt'."
        ) from error

    required = ["CLICKHOUSE_HOST", "CLICKHOUSE_USER", "CLICKHOUSE_PASSWORD"]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(
            "Missing ClickHouse environment variables: "
            + ", ".join(missing)
            + ". Set them before running this script."
        )

    return clickhouse_connect.get_client(
        host=os.environ["CLICKHOUSE_HOST"],
        port=int(os.getenv("CLICKHOUSE_PORT", "8123")),
        username=os.environ["CLICKHOUSE_USER"],
        password=os.environ["CLICKHOUSE_PASSWORD"],
        database=os.getenv("CLICKHOUSE_DATABASE", "default"),
        secure=os.getenv("CLICKHOUSE_SECURE", "false").lower() == "true",
    )


def load_csv_tables(table_names=None):
    #Load requested raw CSV tables without changing source files.#
    names = table_names or TABLE_FILES
    tables = {}
    for table_name in names:
        if table_name not in TABLE_FILES:
            raise ValueError(f"Unknown table: {table_name}")
        csv_path = DATA_DIR / TABLE_FILES[table_name]
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing local CSV source: {csv_path}")
        tables[table_name] = pd.read_csv(csv_path, encoding="utf-8")
    return tables


def select_source(table_names=None):
    #Prefer ClickHouse when configured; otherwise load requested CSV tables.#
    required = ["CLICKHOUSE_HOST", "CLICKHOUSE_USER", "CLICKHOUSE_PASSWORD"]
    missing = [name for name in required if not os.getenv(name)]
    source = os.getenv("EDA_SOURCE", "auto").lower()
    if source == "csv" or (source == "auto" and missing):
        if missing:
            print("ClickHouse variables are not configured; using data/raw CSV files.")
        return "csv", load_csv_tables(table_names)
    if source not in {"auto", "clickhouse"}:
        raise RuntimeError("EDA_SOURCE must be auto, clickhouse, or csv.")
    return "clickhouse", create_client()


def output_path(name):
    ##Resolve an output path relative to the project output directory.
    path = OUTPUT_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path