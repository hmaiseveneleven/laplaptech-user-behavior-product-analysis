"""Data overview: tables, row counts, columns, types, and grain."""

from pathlib import Path

import pandas as pd

from utils import OUTPUT_DIR, quote_identifier, select_source


TABLE_GRAIN = {
    "brand": "one row per brand",
    "cpu_model": "one row per CPU model",
    "gpu_model": "one row per GPU model",
    "laptop_model": "one row per laptop model",
    "laptop_benchmark_result": "one row per laptop benchmark result",
    "user_event_tracking": "one row per tracked user event",
}


def overview_from_csv(tables):
    table_rows = []
    column_rows = []
    for table_name, frame in tables.items():
        table_rows.append(
            {
                "table": table_name,
                "row_count": len(frame),
                "column_count": len(frame.columns),
                "grain": TABLE_GRAIN.get(table_name, "not documented"),
            }
        )
        for position, column in enumerate(frame.columns, start=1):
            column_rows.append(
                {
                    "table": table_name,
                    "ordinal_position": position,
                    "column": column,
                    "data_type": str(frame[column].dtype),
                }
            )
    return pd.DataFrame(table_rows), pd.DataFrame(column_rows)


def overview_from_clickhouse(client):
    table_rows = []
    column_rows = []
    for table_name, grain in TABLE_GRAIN.items():
        table = quote_identifier(table_name)
        count = client.query_df(f"SELECT count() AS row_count FROM {table}").iloc[0]["row_count"]
        schema = client.query_df(
            "SELECT name AS column, position AS ordinal_position, type AS data_type "
            "FROM system.columns WHERE database = currentDatabase() "
            f"AND table = '{table_name}' ORDER BY position"
        )
        table_rows.append(
            {
                "table": table_name,
                "row_count": int(count),
                "column_count": len(schema),
                "grain": grain,
            }
        )
        schema.insert(0, "table", table_name)
        column_rows.extend(schema.to_dict("records"))
    return pd.DataFrame(table_rows), pd.DataFrame(column_rows)


def main():
    source_type, source = select_source()
    if source_type == "csv":
        table_overview, column_schema = overview_from_csv(source)
    else:
        table_overview, column_schema = overview_from_clickhouse(source)
    output_dir = OUTPUT_DIR / "data_overview"
    output_dir.mkdir(parents=True, exist_ok=True)
    table_overview.to_csv(output_dir / "table_overview.csv", index=False)
    column_schema.to_csv(output_dir / "column_schema.csv", index=False)
    print(table_overview.to_string(index=False))
    print(f"Saved: {output_dir}")


if __name__ == "__main__":
    main()