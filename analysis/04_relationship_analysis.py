"""Relationship analysis between laptop attributes and website interactions."""

import json
import re
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from utils import OUTPUT_DIR, select_source


EVENT_SAMPLE_LIMIT = 5000
DETAIL_URL_PATTERN = re.compile(r"/device/(\d+)(?:/|[?#]|$)")


def parse_json_object(payload):
    """Return a JSON object or None for empty, invalid, or non-object payloads."""
    if pd.isna(payload) or not str(payload).strip():
        return None
    try:
        value = json.loads(str(payload))
    except (TypeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def normalize_id(value):
    """Convert integer-like product IDs to int without guessing other formats."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or not re.fullmatch(r"\d+", text):
        return None
    return int(text)


def inspect_event_data_from_csv(frame):
    """Inspect event payload keys and representative samples."""
    key_counts = Counter()
    event_counts = Counter()
    samples = defaultdict(list)
    invalid_json = 0
    for event_name, payload in zip(frame["event_name"], frame["event_data"]):
        event_name = str(event_name)
        event_counts[event_name] += 1
        parsed = parse_json_object(payload)
        if parsed is None:
            if not pd.isna(payload) and str(payload).strip():
                invalid_json += 1
            continue
        for key in parsed:
            key_counts[(event_name, key)] += 1
            if len(samples[(event_name, key)]) < 3:
                samples[(event_name, key)].append(str(payload)[:500])
    schema_rows = [
        {
            "event_name": event_name,
            "event_count": event_counts[event_name],
            "json_key": key,
            "key_count": count,
            "sample_event_data": " || ".join(samples[(event_name, key)]),
        }
        for (event_name, key), count in sorted(key_counts.items())
    ]
    return pd.DataFrame(schema_rows), event_counts, invalid_json


def inspect_event_data_from_clickhouse(client):
    """Inspect a bounded ClickHouse sample without assuming payload semantics."""
    events = client.query_df(
        "SELECT event_name, event_data FROM `user_event_tracking` "
        f"LIMIT {EVENT_SAMPLE_LIMIT}"
    )
    return inspect_event_data_from_csv(events)


def interaction_type(event_name):
    """Map only observed product interaction events to analysis categories."""
    return {
        "pageview": "detail_pageview",
        "select_device_for_comparison": "comparison_select",
        "add_to_comparison": "comparison_add",
        "comparison_chart_sort_selection": "comparison_chart",
    }.get(event_name, "other_mapped_event")


def extract_event_references(frame, model_ids):
    """Normalize confirmed event payload references to laptop IDs.

    Only scalar device_id, device_ids arrays, and pageview detail URLs are used.
    Browser metadata in the separate device column is deliberately ignored.
    """
    records = []
    event_counts = Counter()
    stats = Counter(
        {
            "total_event_rows": len(frame),
            "rows_with_valid_json": 0,
            "scalar_device_id_rows": 0,
            "device_ids_array_rows": 0,
            "pageview_url_device_rows": 0,
            "explicit_url_conflict_rows": 0,
            "non_numeric_id_values": 0,
            "invalid_json_payloads": 0,
            "orphan_laptop_id_references": 0,
            "mapped_event_rows": 0,
            "event_rows_with_mapped_id": 0,
            "mapping_coverage_percent": 0.0,
        }
    )
    invalid_json = 0

    for row in frame.itertuples(index=False):
        payload = parse_json_object(row.event_data)
        if payload is None:
            if not pd.isna(row.event_data) and str(row.event_data).strip():
                invalid_json += 1
            continue
        stats["rows_with_valid_json"] += 1
        event_name = str(row.event_name)
        event_counts[event_name] += 1

        explicit_values = []
        if "device_id" in payload:
            stats["scalar_device_id_rows"] += 1
            explicit_values.append(payload.get("device_id"))
        if "device_ids" in payload:
            stats["device_ids_array_rows"] += 1
            array_values = payload.get("device_ids")
            if isinstance(array_values, list):
                explicit_values.extend(dict.fromkeys(array_values))
            elif array_values is not None:
                stats["non_numeric_id_values"] += 1

        explicit_ids = []
        for value in explicit_values:
            normalized = normalize_id(value)
            if normalized is None and value is not None:
                stats["non_numeric_id_values"] += 1
            elif normalized is not None:
                explicit_ids.append(normalized)
        explicit_ids = list(dict.fromkeys(explicit_ids))

        url_id = None
        if event_name == "pageview":
            match = DETAIL_URL_PATTERN.search(str(payload.get("url", "")))
            if match:
                stats["pageview_url_device_rows"] += 1
                url_id = int(match.group(1))

        if explicit_ids and url_id is not None and explicit_ids[0] != url_id:
            stats["explicit_url_conflict_rows"] += 1
            continue

        references = [(value, "event_data.device_id") for value in explicit_ids]
        if not references and url_id is not None:
            references = [(url_id, "pageview.url")]
        if not references:
            continue

        for laptop_id, mapping_method in references:
            mapping_status = "mapped" if laptop_id in model_ids else "orphan"
            if mapping_status == "orphan":
                stats["orphan_laptop_id_references"] += 1
            records.append(
                {
                    "event_id": getattr(row, "id", None),
                    "event_name": event_name,
                    "user_id": getattr(row, "user_id", None),
                    "session_id": getattr(row, "session_id", None),
                    "event_local_timestamp": getattr(
                        row, "event_local_timestamp", None
                    ),
                    "laptop_model_id": laptop_id,
                    "mapping_method": mapping_method,
                    "mapping_status": mapping_status,
                    "interaction_type": interaction_type(event_name),
                }
            )

    stats["invalid_json_payloads"] = invalid_json
    stats["mapped_event_rows"] = sum(
        1 for record in records if record["mapping_status"] == "mapped"
    )
    mapped_event_ids = {
        record["event_id"]
        for record in records
        if record["mapping_status"] == "mapped"
    }
    stats["event_rows_with_mapped_id"] = len(mapped_event_ids)
    stats["mapping_coverage_percent"] = (
        stats["event_rows_with_mapped_id"] / len(frame) * 100 if len(frame) else 0.0
    )
    return pd.DataFrame(records), stats, event_counts


def load_relationship_data():
    """Load events and dimensions required by all relationship analyses."""
    table_names = [
        "user_event_tracking",
        "laptop_model",
        "laptop_benchmark_result",
        "brand",
    ]
    source_type, source = select_source(table_names)
    if source_type == "csv":
        return source_type, source
    tables = {
        "user_event_tracking": source.query_df(
            "SELECT id, event_name, user_id, event_data, "
            "event_local_timestamp, session_id FROM `user_event_tracking`"
        ),
        "laptop_model": source.query_df(
            "SELECT id, name, is_gaming_laptop, brand_id FROM `laptop_model`"
        ),
        "laptop_benchmark_result": source.query_df(
            "SELECT laptop_model_id, office_battery_result_minutes, "
            "gaming_battery_result_minutes, "
            "geekbench_6_cpu_single_core_plugged_in, "
            "geekbench_6_cpu_multi_core_plugged_in, "
            "geekbench_6_compute_gpu_plugged_in, "
            "geekbench_6_cpu_single_core_battery, "
            "geekbench_6_cpu_multi_core_battery, "
            "geekbench_6_compute_gpu_battery, is_active, changed_on "
            "FROM `laptop_benchmark_result`"
        ),
        "brand": source.query_df("SELECT id, name FROM `brand`")
    }
    return source_type, tables


def prepare_benchmarks(benchmarks):
    """Select one deterministic benchmark row per laptop model."""
    frame = benchmarks.copy()
    frame["laptop_model_id"] = pd.to_numeric(
        frame["laptop_model_id"], errors="coerce"
    )
    frame["is_active_sort"] = (
        frame["is_active"].astype("string").str.lower().eq("true")
    )
    frame["changed_on_sort"] = pd.to_datetime(frame["changed_on"], errors="coerce")
    frame = frame.sort_values(
        ["laptop_model_id", "is_active_sort", "changed_on_sort"],
        ascending=[True, False, False],
    )
    return frame.drop_duplicates("laptop_model_id", keep="first").drop(
        columns=["is_active_sort", "changed_on_sort"], errors="ignore"
    )


def build_model_interactions(events, laptop_models):
    """Aggregate mapped event references to one row per laptop model."""
    model_ids = set(
        pd.to_numeric(laptop_models["id"], errors="coerce").dropna().astype(int)
    )
    normalized, mapping_stats, event_counts = extract_event_references(
        events, model_ids
    )
    mapped = normalized[normalized["mapping_status"] == "mapped"].copy()
    mapped["laptop_model_id"] = mapped["laptop_model_id"].astype(int)
    if mapped.empty:
        aggregates = pd.DataFrame(columns=["laptop_model_id", "total_interactions"])
    else:
        aggregates = mapped.groupby("laptop_model_id").agg(
            total_interactions=("laptop_model_id", "size"),
            detail_pageviews=(
                "interaction_type",
                lambda values: (values == "detail_pageview").sum(),
            ),
            comparison_selects=(
                "interaction_type",
                lambda values: (values == "comparison_select").sum(),
            ),
            comparison_adds=(
                "interaction_type",
                lambda values: (values == "comparison_add").sum(),
            ),
            comparison_chart_participation=(
                "interaction_type",
                lambda values: (values == "comparison_chart").sum(),
            ),
            engaged_sessions=(
                "session_id",
                lambda values: values.dropna().nunique(),
            ),
        ).reset_index()
        aggregates["comparison_interactions"] = (
            aggregates["comparison_selects"]
            + aggregates["comparison_adds"]
            + aggregates["comparison_chart_participation"]
        )
    model_frame = laptop_models.copy()
    model_frame["laptop_model_id"] = pd.to_numeric(
        model_frame["id"], errors="coerce"
    )
    model_frame = model_frame.rename(columns={"id": "laptop_id"})
    model_frame = model_frame.merge(aggregates, on="laptop_model_id", how="left")
    count_columns = [
        "total_interactions",
        "detail_pageviews",
        "comparison_selects",
        "comparison_adds",
        "comparison_chart_participation",
        "engaged_sessions",
        "comparison_interactions",
    ]
    model_frame[count_columns] = model_frame[count_columns].fillna(0).astype(int)
    model_frame["interactions_per_laptop"] = model_frame["total_interactions"]
    return model_frame, normalized, mapping_stats, event_counts


def write_mapping_outputs(schema, event_counts, mapping_stats, normalized, output_dir):
    """Write auditable event inspection and mapping QA reports."""
    schema.to_csv(output_dir / "event_data_schema.csv", index=False)
    pd.DataFrame(
        [
            {"event_name": name, "event_count": count}
            for name, count in sorted(event_counts.items())
        ]
    ).to_csv(output_dir / "event_name_summary.csv", index=False)
    qa_columns = [
        "total_event_rows",
        "rows_with_valid_json",
        "scalar_device_id_rows",
        "device_ids_array_rows",
        "pageview_url_device_rows",
        "explicit_url_conflict_rows",
        "non_numeric_id_values",
        "invalid_json_payloads",
        "orphan_laptop_id_references",
        "mapped_event_rows",
        "event_rows_with_mapped_id",
        "mapping_coverage_percent",
    ]
    pd.DataFrame([dict(mapping_stats)]).reindex(columns=qa_columns).to_csv(
        output_dir / "mapping_status.csv", index=False
    )
    if normalized.empty:
        by_event = pd.DataFrame(
            columns=[
                "event_name",
                "mapping_method",
                "mapping_status",
                "reference_count",
            ]
        )
    else:
        by_event = (
            normalized.groupby(
                ["event_name", "mapping_method", "mapping_status"]
            )
            .size()
            .reset_index(name="reference_count")
        )
    by_event.to_csv(output_dir / "mapping_coverage_by_event.csv", index=False)


def save_scatter(frame, x_column, y_column, title, path):
    """Save a scatter plot only when both axes have usable values."""
    points = frame[[x_column, y_column]].dropna()
    if len(points) < 2:
        return False
    plt.figure(figsize=(7, 4.5))
    plt.scatter(points[x_column], points[y_column], alpha=0.65)
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return True


def correlation_rows(frame, metric_columns):
    """Calculate correlations only from complete pairs."""
    rows = []
    for metric in metric_columns:
        points = frame[[metric, "total_interactions"]].dropna()
        rows.append(
            {
                "metric": metric,
                "observations": len(points),
                "pearson_correlation": points[metric].corr(
                    points["total_interactions"]
                )
                if len(points) >= 2
                else None,
                "spearman_correlation": points[metric]
                .rank(method="average")
                .corr(points["total_interactions"].rank(method="average"))
                if len(points) >= 2
                else None,
                "analysis_possible": len(points) >= 2,
            }
        )
    return pd.DataFrame(rows)


def analyze_gaming_vs_nongaming(model_frame, output_dir):
    """Compare mapped interaction volume between gaming categories."""
    frame = model_frame.copy()
    frame["gaming_group"] = frame["is_gaming_laptop"].astype("string").str.lower()
    frame["gaming_group"] = frame["gaming_group"].replace(
        {"true": "gaming", "false": "non_gaming", "<na>": "unknown"}
    )
    result = frame.groupby("gaming_group", dropna=False).agg(
        laptop_count=("laptop_model_id", "nunique"),
        laptops_with_interactions=(
            "total_interactions",
            lambda values: (values > 0).sum(),
        ),
        total_interactions=("total_interactions", "sum"),
        detail_pageviews=("detail_pageviews", "sum"),
        comparison_interactions=("comparison_interactions", "sum"),
        engaged_sessions=("engaged_sessions", "sum"),
        average_interactions_per_laptop=("total_interactions", "mean"),
    ).reset_index()
    result.to_csv(output_dir / "gaming_vs_nongaming.csv", index=False)
    result.plot(
        x="gaming_group",
        y="total_interactions",
        kind="bar",
        legend=False,
        title="Total mapped interactions by laptop type",
    )
    plt.tight_layout()
    plt.savefig(output_dir / "gaming_vs_nongaming.png", dpi=150)
    plt.close()
    return result


def analyze_battery_vs_interaction(model_frame, benchmarks, output_dir):
    """Relate battery measures without imputing missing values."""
    metrics = ["office_battery_result_minutes", "gaming_battery_result_minutes"]
    frame = model_frame.merge(
        prepare_benchmarks(benchmarks), on="laptop_model_id", how="left"
    )
    for metric in metrics:
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    output_columns = ["laptop_model_id", "name", "total_interactions"] + metrics
    frame[output_columns].to_csv(output_dir / "battery_vs_interaction.csv", index=False)
    correlations = correlation_rows(frame, metrics)
    correlations.to_csv(output_dir / "battery_correlation_summary.csv", index=False)
    for metric in metrics:
        save_scatter(
            frame,
            metric,
            "total_interactions",
            f"{metric} vs total interactions",
            output_dir / f"{metric}_vs_interaction.png",
        )
    return frame, correlations


def analyze_geekbench_vs_interaction(model_frame, benchmarks, output_dir):
    """Relate available Geekbench measures to interactions."""
    metrics = [
        "geekbench_6_cpu_single_core_plugged_in",
        "geekbench_6_cpu_multi_core_plugged_in",
        "geekbench_6_compute_gpu_plugged_in",
        "geekbench_6_cpu_single_core_battery",
        "geekbench_6_cpu_multi_core_battery",
        "geekbench_6_compute_gpu_battery",
    ]
    frame = model_frame.merge(
        prepare_benchmarks(benchmarks), on="laptop_model_id", how="left"
    )
    for metric in metrics:
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    output_columns = ["laptop_model_id", "name", "total_interactions"] + metrics
    frame[output_columns].to_csv(output_dir / "geekbench_vs_interaction.csv", index=False)
    correlations = correlation_rows(frame, metrics)
    correlations.to_csv(output_dir / "geekbench_correlation_summary.csv", index=False)
    for metric in metrics:
        save_scatter(
            frame,
            metric,
            "total_interactions",
            f"{metric} vs total interactions",
            output_dir / f"{metric}_vs_interaction.png",
        )
    return frame, correlations


def analyze_brand_vs_interaction(model_frame, brands, output_dir):
    """Aggregate mapped interactions by the confirmed brand foreign key."""
    frame = model_frame.copy()
    frame["brand_id"] = pd.to_numeric(frame["brand_id"], errors="coerce")
    brand_frame = brands.rename(columns={"id": "brand_id", "name": "brand_name"})
    frame = frame.merge(brand_frame, on="brand_id", how="left")
    result = frame.groupby("brand_name", dropna=False).agg(
        laptop_count=("laptop_model_id", "nunique"),
        laptops_with_interactions=(
            "total_interactions",
            lambda values: (values > 0).sum(),
        ),
        total_interactions=("total_interactions", "sum"),
        detail_pageviews=("detail_pageviews", "sum"),
        engaged_sessions=("engaged_sessions", "sum"),
        average_interactions_per_laptop=("total_interactions", "mean"),
        median_interactions_per_laptop=("total_interactions", "median"),
    ).reset_index()
    result["interaction_rate_per_laptop"] = (
        result["total_interactions"] / result["laptop_count"]
    )
    result.to_csv(output_dir / "brand_vs_interaction.csv", index=False)
    result.sort_values("total_interactions").plot(
        x="brand_name",
        y="total_interactions",
        kind="barh",
        legend=False,
        title="Total mapped interactions by brand",
    )
    plt.tight_layout()
    plt.savefig(output_dir / "brand_interaction_bar.png", dpi=150)
    plt.close()
    return result


def analyze_high_benchmark_high_interaction(model_frame, benchmarks, output_dir):
    """Identify high/high laptops using Q3 thresholds."""
    frame = model_frame.merge(
        prepare_benchmarks(benchmarks), on="laptop_model_id", how="left"
    )
    metric = "geekbench_6_cpu_multi_core_plugged_in"
    frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    valid = frame[[metric, "total_interactions"]].dropna()
    if valid.empty:
        frame["high_benchmark"] = False
        frame["high_interaction"] = False
        frame["quadrant"] = "insufficient_data"
        thresholds = {"benchmark_q3": None, "interaction_q3": None}
    else:
        benchmark_q3 = valid[metric].quantile(0.75)
        interaction_q3 = valid["total_interactions"].quantile(0.75)
        frame["high_benchmark"] = frame[metric] >= benchmark_q3
        frame["high_interaction"] = frame["total_interactions"] >= interaction_q3
        frame["quadrant"] = "low_benchmark_low_interaction"
        frame.loc[
            frame["high_benchmark"] & ~frame["high_interaction"], "quadrant"
        ] = "high_benchmark_low_interaction"
        frame.loc[
            ~frame["high_benchmark"] & frame["high_interaction"], "quadrant"
        ] = "low_benchmark_high_interaction"
        frame.loc[
            frame["high_benchmark"] & frame["high_interaction"], "quadrant"
        ] = "high_benchmark_high_interaction"
        thresholds = {
            "benchmark_q3": benchmark_q3,
            "interaction_q3": interaction_q3,
        }
    columns = ["laptop_model_id", "name", metric, "total_interactions", "quadrant"]
    frame[columns].to_csv(output_dir / "high_benchmark_high_interaction.csv", index=False)
    pd.DataFrame([thresholds]).to_csv(
        output_dir / "high_high_thresholds.csv", index=False
    )
    points = frame[[metric, "total_interactions"]].dropna()
    if len(points) >= 2:
        points.plot.scatter(
            x=metric,
            y="total_interactions",
            title="Benchmark vs interactions",
        )
        plt.tight_layout()
        plt.savefig(output_dir / "high_benchmark_high_interaction.png", dpi=150)
        plt.close()
    return frame


def main():
    source_type, tables = load_relationship_data()
    output_dir = OUTPUT_DIR / "relationship_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    events = tables["user_event_tracking"]
    schema, event_counts, _ = inspect_event_data_from_csv(
        events[["event_name", "event_data"]]
    )
    model_frame, normalized, mapping_stats, _ = build_model_interactions(
        events, tables["laptop_model"]
    )
    write_mapping_outputs(
        schema, event_counts, mapping_stats, normalized, output_dir
    )
    analyze_gaming_vs_nongaming(model_frame, output_dir)
    _, battery_correlations = analyze_battery_vs_interaction(
        model_frame, tables["laptop_benchmark_result"], output_dir
    )
    _, geekbench_correlations = analyze_geekbench_vs_interaction(
        model_frame, tables["laptop_benchmark_result"], output_dir
    )
    brand_result = analyze_brand_vs_interaction(
        model_frame, tables["brand"], output_dir
    )
    high_high = analyze_high_benchmark_high_interaction(
        model_frame, tables["laptop_benchmark_result"], output_dir
    )
    summary = pd.DataFrame(
        [
            {"item": "source", "value": source_type},
            {"item": "total_event_rows", "value": mapping_stats["total_event_rows"]},
            {"item": "mapped_event_references", "value": mapping_stats["mapped_event_rows"]},
            {"item": "mapping_coverage_percent", "value": mapping_stats["mapping_coverage_percent"]},
            {"item": "battery_correlations_available", "value": int(battery_correlations["analysis_possible"].sum())},
            {"item": "geekbench_correlations_available", "value": int(geekbench_correlations["analysis_possible"].sum())},
            {"item": "brand_rows", "value": len(brand_result)},
            {"item": "high_high_rows", "value": int((high_high["quadrant"] == "high_benchmark_high_interaction").sum())},
        ]
    )
    summary.to_csv(output_dir / "relationship_summary.csv", index=False)
    print(f"Source: {source_type}")
    print(f"Mapped event references: {mapping_stats['mapped_event_rows']}")
    print(f"Mapping coverage: {mapping_stats['mapping_coverage_percent']:.2f}%")
    print(f"Saved relationship analysis files to: {output_dir}")


if __name__ == "__main__":
    main()
