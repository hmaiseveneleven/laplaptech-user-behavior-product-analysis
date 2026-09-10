"""Data quality checks for missing values, duplicates, keys, and formats."""

import re

import pandas as pd

from utils import OUTPUT_DIR, quote_identifier, select_source


PRIMARY_KEYS = {
    "brand": "id",
    "cpu_model": "id",
    "gpu_model": "id",
    "laptop_model": "id",
    "laptop_benchmark_result": "id",
    "user_event_tracking": "id",
}

FOREIGN_KEYS = [
    ("cpu_model", "brand_id", "brand", "id"),
    ("gpu_model", "brand_id", "brand", "id"),
    ("laptop_model", "brand_id", "brand", "id"),
    ("laptop_model", "cpu_model_id", "cpu_model", "id"),
    ("laptop_model", "gpu_model_id", "gpu_model", "id"),
    ("laptop_benchmark_result", "laptop_model_id", "laptop_model", "id"),
]

TDP_FIELDS = ("cpu_tdp", "gpu_tdp", "frequency")
WEIGHT_DETAIL_FIELDS = (
    "id",
    "name",
    "laptop_weight",
    "brand_id",
    "screen_size",
    "is_gaming_laptop",
)


def check_missing_values(tables):
    rows = []
    for table_name, frame in tables.items():
        for column in frame.columns:
            null_count = int(frame[column].isna().sum())
            empty_count = int(
                frame[column].astype("string").str.strip().eq("").sum()
            )
            rows.append(
                {
                    "table": table_name,
                    "column": column,
                    "null_count": null_count,
                    "empty_count": empty_count,
                    "total_count": len(frame),
                    "missing_percent": (null_count + empty_count) / len(frame) * 100
                    if len(frame)
                    else 0.0,
                }
            )
    return pd.DataFrame(rows)


def check_duplicates(tables):
    rows = []
    for table_name, frame in tables.items():
        duplicate_rows = int(frame.duplicated(keep=False).sum())
        rows.append(
            {
                "table": table_name,
                "duplicate_row_count": duplicate_rows,
                "duplicate_group_count": int(frame.duplicated().sum()),
            }
        )
    return pd.DataFrame(rows)


def check_primary_keys(tables):
    rows = []
    for table_name, key in PRIMARY_KEYS.items():
        values = tables[table_name][key]
        rows.append(
            {
                "table": table_name,
                "column": key,
                "null_count": int(values.isna().sum()),
                "duplicate_count": int(values.duplicated(keep=False).sum()),
                "unique": bool(values.notna().all() and values.is_unique),
            }
        )
    return pd.DataFrame(rows)


def check_foreign_keys(tables):
    rows = []
    for child_table, child_key, parent_table, parent_key in FOREIGN_KEYS:
        child_values = tables[child_table][child_key].dropna()
        parent_values = set(tables[parent_table][parent_key].dropna())
        orphan_count = int((~child_values.isin(parent_values)).sum())
        rows.append(
            {
                "child_table": child_table,
                "child_column": child_key,
                "parent_table": parent_table,
                "parent_column": parent_key,
                "non_null_child_count": len(child_values),
                "orphan_count": orphan_count,
                "integrity_ok": orphan_count == 0,
            }
        )
    return pd.DataFrame(rows)


def check_datatypes(tables):
    rows = []
    for table_name, frame in tables.items():
        for column in frame.columns:
            rows.append(
                {
                    "table": table_name,
                    "column": column,
                    "pandas_dtype": str(frame[column].dtype),
                    "numeric_parse_success_percent": pd.to_numeric(
                        frame[column], errors="coerce"
                    ).notna().mean()
                    * 100,
                }
            )
    return pd.DataFrame(rows)


def classify_tdp_value(value):
    if pd.isna(value):
        return "null"
    text = str(value).strip()
    if not text:
        return "empty"
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", text):
        return "numeric_only"
    if re.search(r"[wW]", text):
        return "contains_w"
    return "other"


def check_tdp_formats(tables):
    """Return distinct raw TDP values and their observed format categories."""
    rows = []
    for table_name, frame in tables.items():
        for column in TDP_FIELDS:
            if column not in frame.columns:
                continue
            values = frame[column].map(lambda value: "" if pd.isna(value) else str(value))
            grouped = (
                pd.DataFrame(
                    {
                        "raw_value": values,
                        "format_category": frame[column].map(classify_tdp_value),
                    }
                )
                .groupby(["raw_value", "format_category"], dropna=False)
                .size()
                .reset_index(name="value_count")
            )
            grouped.insert(0, "table", table_name)
            grouped.insert(1, "column", column)
            rows.extend(grouped.to_dict("records"))
    return pd.DataFrame(
        rows,
        columns=["table", "column", "raw_value", "format_category", "value_count"],
    )


def check_tdp_schema(tables):
    """Record whether requested TDP-related columns exist in the loaded schema."""
    rows = []
    for table_name, frame in tables.items():
        for column in TDP_FIELDS:
            rows.append(
                {
                    "table": table_name,
                    "column": column,
                    "present": column in frame.columns,
                }
            )
    return pd.DataFrame(rows)


def check_laptop_weight_distribution(tables):
    """Summarize laptop weight values without inferring or changing units."""
    frame = tables["laptop_model"]
    values = pd.to_numeric(frame["laptop_weight"], errors="coerce")
    numeric_values = values.dropna()
    stats = {
        "table": "laptop_model",
        "column": "laptop_weight",
        "count": len(frame),
        "null_count": int(frame["laptop_weight"].isna().sum()),
        "non_numeric_count": int(frame["laptop_weight"].notna().sum() - values.notna().sum()),
        "numeric_count": int(values.notna().sum()),
        "min": numeric_values.min() if not numeric_values.empty else None,
        "max": numeric_values.max() if not numeric_values.empty else None,
        "mean": numeric_values.mean() if not numeric_values.empty else None,
        "median": numeric_values.median() if not numeric_values.empty else None,
        "q1": numeric_values.quantile(0.25) if not numeric_values.empty else None,
        "q3": numeric_values.quantile(0.75) if not numeric_values.empty else None,
        "p90": numeric_values.quantile(0.90) if not numeric_values.empty else None,
        "p95": numeric_values.quantile(0.95) if not numeric_values.empty else None,
        "p99": numeric_values.quantile(0.99) if not numeric_values.empty else None,
    }
    return pd.DataFrame([stats])


def check_laptop_weight_groups(tables):
    """Count requested numeric laptop-weight ranges, including nulls."""
    values = pd.to_numeric(tables["laptop_model"]["laptop_weight"], errors="coerce")
    groups = pd.Series("null_or_non_numeric", index=values.index, dtype="string")
    groups.loc[values == 0] = "equal_0"
    groups.loc[(values > 0) & (values < 10)] = "greater_0_less_10"
    groups.loc[(values >= 10) & (values < 100)] = "greater_equal_10_less_100"
    groups.loc[values >= 100] = "greater_equal_100"
    return groups.value_counts(dropna=False).rename_axis("weight_group").reset_index(name="row_count")


def check_suspicious_laptop_weights(tables):
    """Return source details for zero or at least 10 weight values."""
    laptop = tables["laptop_model"].copy()
    values = pd.to_numeric(laptop["laptop_weight"], errors="coerce")
    mask = values.notna() & ((values == 0) | (values >= 10))
    result = laptop.loc[mask, [field for field in WEIGHT_DETAIL_FIELDS if field in laptop.columns]].copy()
    result.insert(0, "suspicion_reason", values.loc[mask].map(
        lambda value: "equal_0" if value == 0 else "greater_equal_10"
    ).to_numpy())
    if "brand_id" in result.columns and "brand" in tables:
        brands = tables["brand"]
        if {"id", "name"}.issubset(brands.columns):
            brand_names = brands.set_index("id")["name"]
            result.insert(
                result.columns.get_loc("brand_id") + 1,
                "brand",
                result["brand_id"].map(brand_names),
            )
    return result


def load_clickhouse_tables(client):
    """Read the six source tables for non-destructive quality checks."""
    return {
        table_name: client.query_df(
            f"SELECT * FROM {quote_identifier(table_name)}"
        )
        for table_name in PRIMARY_KEYS
    }


def main():
    source_type, source = select_source()
    tables = source if source_type == "csv" else load_clickhouse_tables(source)
    output_dir = OUTPUT_DIR / "data_quality"
    output_dir.mkdir(parents=True, exist_ok=True)
    checks = {
        "missing_values.csv": check_missing_values(tables),
        "duplicate_summary.csv": check_duplicates(tables),
        "pk_check.csv": check_primary_keys(tables),
        "fk_check.csv": check_foreign_keys(tables),
        "datatype_check.csv": check_datatypes(tables),
        "tdp_format_distribution.csv": check_tdp_formats(tables),
        "tdp_schema_check.csv": check_tdp_schema(tables),
        "laptop_weight_statistics.csv": check_laptop_weight_distribution(tables),
        "laptop_weight_groups.csv": check_laptop_weight_groups(tables),
        "suspicious_laptop_weights.csv": check_suspicious_laptop_weights(tables),
    }
    for filename, frame in checks.items():
        frame.to_csv(output_dir / filename, index=False)
        print(f"Saved: {output_dir / filename}")


if __name__ == "__main__":
    main()