"""Detect numeric outliers with the IQR method."""

# Outliers are reported, never deleted or modified. ClickHouse is preferred
# when configured; bundled CSV files are used otherwise.


import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from utils import OUTPUT_DIR, quote_identifier, select_source


NUMERIC_COLUMNS = {
    "LAPTOP_MODEL": [
        "screen_size",
        "screen_dimension_width",
        "screen_dimension_height",
        "screen_ppi",
        "laptop_weight",
        "charger_weight",
        "battery_capacity_whr",
        "year_introduce",
    ],
    "LAPTOP_BENCHMARK_RESULT": [
        "office_battery_result_minutes",
        "gaming_battery_result_minutes",
        "foldable_opening_battery_result_minutes",
        "geekbench_6_compute_gpu_plugged_in",
        "geekbench_6_compute_gpu_battery",
        "geekbench_6_cpu_single_core_plugged_in",
        "geekbench_6_cpu_single_core_battery",
        "geekbench_6_cpu_multi_core_plugged_in",
        "geekbench_6_cpu_multi_core_battery",
    ],
}


def load_outlier_csv_tables():
    """Load only the two source tables needed for outlier analysis."""
    source_type, source = select_source(
        [table_name.lower() for table_name in NUMERIC_COLUMNS]
    )
    if source_type != "csv":
        return source_type, source
    return source_type, {
        table_name: source[table_name.lower()]
        for table_name in NUMERIC_COLUMNS
    }


def fetch_column_stats(client, table_name, column_name):
    """Calculate all IQR statistics in ClickHouse for one numeric column."""
    table = quote_identifier(table_name.lower())
    column = quote_identifier(column_name)
    query = f"""
        WITH values AS (
            SELECT toFloat64OrNull(toString({column})) AS value
            FROM {table}
        ),
        quartiles AS (
            SELECT
                quantile(0.25)(value) AS q1,
                quantile(0.50)(value) AS median,
                quantile(0.75)(value) AS q3
            FROM values
        )
        SELECT
            count(value) AS non_null,
            (SELECT count() FROM {table}) - count(value) AS null_count,
            min(value) AS min,
            q1,
            median,
            q3,
            max(value) AS max,
            q3 - q1 AS iqr,
            q1 - 1.5 * (q3 - q1) AS lower_bound,
            q3 + 1.5 * (q3 - q1) AS upper_bound,
            countIf(value < lower_bound OR value > upper_bound) AS outlier_count
        FROM values
        CROSS JOIN quartiles
    """
    result = client.query_df(query)
    if result.empty:
        return None

    row = result.iloc[0].to_dict()
    row["table"] = table_name
    row["column"] = column_name
    row["non_null"] = int(row["non_null"] or 0)
    row["null_count"] = int(row["null_count"] or 0)
    row["null_percent"] = (
        row["null_count"] / (row["non_null"] + row["null_count"]) * 100
        if row["non_null"] + row["null_count"]
        else 0.0
    )
    row["outlier_count"] = int(row["outlier_count"] or 0)
    row["outlier_percent"] = (
        row["outlier_count"] / row["non_null"] * 100
        if row["non_null"]
        else 0.0
    )
    return row


def fetch_csv_column_stats(tables, table_name, column_name):
    """Calculate IQR statistics from a local CSV without modifying source data."""
    values = pd.to_numeric(tables[table_name][column_name], errors="coerce")
    non_null_values = values.dropna()
    if non_null_values.empty:
        return None

    q1 = non_null_values.quantile(0.25)
    median = non_null_values.quantile(0.50)
    q3 = non_null_values.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_mask = (values < lower_bound) | (values > upper_bound)
    outlier_count = int(outlier_mask.sum())
    non_null = int(non_null_values.size)
    null_count = int(values.size - non_null)
    return {
        "table": table_name,
        "column": column_name,
        "non_null": non_null,
        "null_count": null_count,
        "null_percent": null_count / len(values) * 100 if len(values) else 0.0,
        "min": non_null_values.min(),
        "q1": q1,
        "median": median,
        "q3": q3,
        "max": non_null_values.max(),
        "iqr": iqr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outlier_count": outlier_count,
        "outlier_percent": outlier_count / non_null * 100 if non_null else 0.0,
    }


def fetch_outlier_records(client, stat):
    """Fetch only IDs and values beyond the IQR bounds."""
    if stat["outlier_count"] == 0:
        return []

    table = quote_identifier(stat["table"].lower())
    column = quote_identifier(stat["column"])
    query = f"""
        SELECT toString(id) AS id, toFloat64OrNull(toString({column})) AS value
        FROM {table}
        WHERE {column} IS NOT NULL
          AND (toFloat64OrNull(toString({column})) < {{lower:Float64}}
               OR toFloat64OrNull(toString({column})) > {{upper:Float64}})
        ORDER BY value
    """
    result = client.query_df(
        query,
        parameters={"lower": stat["lower_bound"], "upper": stat["upper_bound"]},
    )
    return [
        {
            "table": stat["table"],
            "id": row["id"],
            "column": stat["column"],
            "value": row["value"],
        }
        for _, row in result.iterrows()
    ]


def fetch_csv_outlier_records(tables, stat):
    """Return source row IDs and outlier values from a local CSV."""
    table = tables[stat["table"]]
    values = pd.to_numeric(table[stat["column"]], errors="coerce")
    mask = values.notna() & (
        (values < stat["lower_bound"]) | (values > stat["upper_bound"])
    )
    return [
        {
            "table": stat["table"],
            "id": str(row_id),
            "column": stat["column"],
            "value": value,
        }
        for row_id, value in zip(table.loc[mask, "id"], values.loc[mask])
    ]


def fetch_plot_values(client, stat):
    """Fetch non-null values for a boxplot."""
    table = quote_identifier(stat["table"].lower())
    column = quote_identifier(stat["column"])
    result = client.query_df(
        f"SELECT toFloat64OrNull(toString({column})) AS value FROM {table} "
        f"WHERE {column} IS NOT NULL"
    )
    return pd.to_numeric(result["value"], errors="coerce").dropna()


def fetch_csv_plot_values(tables, stat):
    """Return numeric non-null values for a local CSV boxplot."""
    return pd.to_numeric(
        tables[stat["table"]][stat["column"]], errors="coerce"
    ).dropna()


def print_statistic(stat):
    print(f"\nColumn: {stat['column']}")
    print(
        f"Non-null: {stat['non_null']} | Null: {stat['null_count']} "
        f"({stat['null_percent']:.2f}%)"
    )
    for label, key in [
        ("Min", "min"),
        ("Q1", "q1"),
        ("Median", "median"),
        ("Q3", "q3"),
        ("Max", "max"),
        ("IQR", "iqr"),
        ("Lower bound", "lower_bound"),
        ("Upper bound", "upper_bound"),
    ]:
        print(f"{label}: {stat[key]:.4f}")
    print(f"Outlier count: {stat['outlier_count']}")
    print(f"Outlier percent: {stat['outlier_percent']:.2f}%")


def save_boxplot(values, stat, plot_dir):
    """Save a matplotlib boxplot for a column with outliers."""
    plt.figure(figsize=(7, 4.5))
    plt.boxplot(
        values,
        vert=False,
        patch_artist=True,
        boxprops={"facecolor": "#8ecae6"},
    )
    plt.title(f"{stat['table']}: {stat['column']}")
    plt.xlabel("Value")
    plt.grid(axis="x", alpha=0.3)
    output_path = plot_dir / f"{stat['table'].lower()}_{stat['column']}.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved: {output_path}")


def print_interpretation(summary, records):
    """Print data-driven findings without declaring outliers invalid."""
    with_outliers = summary[summary["outlier_count"] > 0]
    print("\n" + "=" * 60)
    print("DATA-DRIVEN INTERPRETATION")
    print("=" * 60)
    print(f"Attributes with outliers: {len(with_outliers)} / {len(summary)}")
    if with_outliers.empty:
        print("No IQR outliers were found; no outlier-specific conclusion is made.")
        return

    most_count = with_outliers.sort_values("outlier_count", ascending=False).iloc[0]
    most_percent = with_outliers.sort_values(
        "outlier_percent", ascending=False
    ).iloc[0]
    print(
        f"Most outliers by count: {most_count['table']}.{most_count['column']} "
        f"({int(most_count['outlier_count'])}, "
        f"{most_count['outlier_percent']:.2f}%)."
    )
    print(
        f"Highest outlier rate: {most_percent['table']}.{most_percent['column']} "
        f"({most_percent['outlier_percent']:.2f}%)."
    )
    print(
        "Potential input/unit errors must be checked against the source definition, "
        "units, and product specifications; this script does not label them as wrong."
    )
    print("Extreme but plausible product specifications may be legitimate rare products.")
    print(
        "For inherently skewed or heterogeneous measurements, do not remove outliers "
        "solely because they exceed an IQR fence."
    )
    print(f"Outlier records exported for review: {len(records)}")
    print(
        "Cleaning recommendation: validate units and domain ranges, investigate source "
        "rows, then decide whether to correct, keep, cap, or exclude with documented rules."
    )


def main():
    source_type, source = load_outlier_csv_tables()
    output_dir = Path(os.getenv("EDA_OUTPUT_DIR", OUTPUT_DIR / "outlier"))
    plot_dir = output_dir / "outlier_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    outlier_records = []

    for table_name, columns in NUMERIC_COLUMNS.items():
        print("\n" + "=" * 60)
        print(f"OUTLIER ANALYSIS - {table_name}")
        print("=" * 60)
        for column_name in columns:
            if source_type == "clickhouse":
                stat = fetch_column_stats(source, table_name, column_name)
            else:
                stat = fetch_csv_column_stats(source, table_name, column_name)
            if stat is None:
                print(f"Column: {column_name} (no result returned)")
                continue
            summary_rows.append(stat)
            print_statistic(stat)
            if source_type == "clickhouse":
                outlier_records.extend(fetch_outlier_records(source, stat))
            else:
                outlier_records.extend(fetch_csv_outlier_records(source, stat))
            if stat["outlier_count"] > 0:
                if source_type == "clickhouse":
                    values = fetch_plot_values(source, stat)
                else:
                    values = fetch_csv_plot_values(source, stat)
                if not values.empty:
                    save_boxplot(values, stat, plot_dir)

    summary_columns = [
        "table",
        "column",
        "non_null",
        "null_count",
        "null_percent",
        "min",
        "q1",
        "median",
        "q3",
        "max",
        "iqr",
        "lower_bound",
        "upper_bound",
        "outlier_count",
        "outlier_percent",
    ]
    summary = pd.DataFrame(summary_rows, columns=summary_columns)
    records = pd.DataFrame(outlier_records, columns=["table", "id", "column", "value"])
    summary.to_csv(output_dir / "outlier_summary.csv", index=False)
    records.to_csv(output_dir / "outlier_records.csv", index=False)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(summary.to_string(index=False))
    print(f"\nSaved: {output_dir / 'outlier_summary.csv'}")
    print(f"Saved: {output_dir / 'outlier_records.csv'}")
    print_interpretation(summary, records)


if __name__ == "__main__":
    main()