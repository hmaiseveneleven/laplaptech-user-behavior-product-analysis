"""Baseline overview of user events, activity, and device metadata."""

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from utils import OUTPUT_DIR, quote_identifier, select_source


EVENT_TABLE = "user_event_tracking"
UNKNOWN_CATEGORY = "(unknown)"


def load_events():
    """Load the event table from the configured source without changing it."""
    source_type, source = select_source([EVENT_TABLE])
    if source_type == "csv":
        return source[EVENT_TABLE]
    return source.query_df(f"SELECT * FROM {quote_identifier(EVENT_TABLE)}")


def non_empty_text(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def event_distribution(frame):
    """Count every row, including missing or empty event names."""
    if "event_name" not in frame.columns:
        return pd.DataFrame(
            columns=["event_name", "event_count", "event_percentage"]
        )
    event_names = frame["event_name"].map(non_empty_text).fillna(UNKNOWN_CATEGORY)
    distribution = event_names.value_counts().rename_axis("event_name").reset_index(
        name="event_count"
    )
    distribution["event_percentage"] = (
        distribution["event_count"] / len(frame) * 100 if len(frame) else 0.0
    )
    return distribution.sort_values(
        ["event_count", "event_name"], ascending=[False, True]
    ).reset_index(drop=True)


def activity_summary(frame, identifier_column, entity_label):
    """Summarize event counts for non-null users or sessions."""
    columns = [
        "entity",
        "entity_count",
        "total_events",
        "mean_events",
        "median_events",
        "min_events",
        "max_events",
        "p90_events",
        "p95_events",
    ]
    if identifier_column not in frame.columns:
        return pd.DataFrame(
            [{"metric": entity_label, "value": None, "status": "not_available"}],
            columns=["metric", "value", "status"],
        )

    identifiers = frame[identifier_column].map(non_empty_text)
    counts = identifiers.dropna().value_counts()
    if counts.empty:
        return pd.DataFrame(
            [{"metric": entity_label, "value": None, "status": "no_non_null_ids"}],
            columns=["metric", "value", "status"],
        )

    return pd.DataFrame(
        [
            {
                "entity": entity_label,
                "entity_count": int(counts.size),
                "total_events": int(counts.sum()),
                "mean_events": counts.mean(),
                "median_events": counts.median(),
                "min_events": counts.min(),
                "max_events": counts.max(),
                "p90_events": counts.quantile(0.90),
                "p95_events": counts.quantile(0.95),
            }
        ],
        columns=columns,
    )


def kpi_row(metric, value, status, note):
    return {"metric": metric, "value": value, "status": status, "note": note}


def kpi_summary(frame, distribution):
    """Build KPI rows while making unavailable fields explicit."""
    rows = [kpi_row("total_events", len(frame), "available", "All source rows")]
    for column, label in (("user_id", "users"), ("session_id", "sessions")):
        if column not in frame.columns:
            rows.append(
                kpi_row(
                    f"unique_{label}",
                    None,
                    "not_available",
                    f"Column {column} is not present",
                )
            )
            rows.append(
                kpi_row(
                    f"events_per_{label[:-1]}",
                    None,
                    "not_available",
                    f"Column {column} is not present",
                )
            )
            continue
        identifiers = frame[column].map(non_empty_text)
        non_null = identifiers.dropna()
        unique_count = int(non_null.nunique())
        rows.append(
            kpi_row(
                f"unique_{label}",
                unique_count,
                "available",
                f"Distinct non-null {column} values",
            )
        )
        activity = non_null.value_counts()
        rows.append(
            kpi_row(
                f"events_per_{label[:-1]}",
                activity.mean() if not activity.empty else None,
                "available" if not activity.empty else "no_non_null_ids",
                "Total events divided across non-null identifiers",
            )
        )

    rows.append(
        kpi_row(
            "event_percentage_total",
            distribution["event_percentage"].sum() if not distribution.empty else 0.0,
            "available",
            "Must be approximately 100%; unknown event names remain included",
        )
    )
    return pd.DataFrame(rows)


def parse_json_object(value):
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def extract_device(frame):
    """Extract a supported device dimension, preserving unknown rows."""
    if "device" not in frame.columns:
        return None

    records = []
    for payload in frame["device"]:
        parsed = parse_json_object(payload)
        if parsed is None:
            records.append((UNKNOWN_CATEGORY, "device"))
            continue
        selected = None
        selected_field = None
        for field in ("device_type", "device", "browser", "os_name"):
            value = non_empty_text(parsed.get(field))
            if value is not None:
                selected = value
                selected_field = field
                break
        records.append((selected or UNKNOWN_CATEGORY, selected_field or "device"))
    return pd.DataFrame(records, columns=["device_category", "device_field"], index=frame.index)


def device_distribution(frame):
    extracted = extract_device(frame)
    if extracted is None:
        return None
    work = frame.join(extracted)
    rows = []
    total_events = len(work)
    for category, group in work.groupby("device_category", dropna=False):
        rows.append(
            {
                "device_category": category,
                "device_field": ", ".join(sorted(group["device_field"].unique())),
                "event_count": len(group),
                "event_percentage": len(group) / total_events * 100
                if total_events
                else 0.0,
                "unique_users": (
                    group["user_id"].map(non_empty_text).dropna().nunique()
                    if "user_id" in group
                    else None
                ),
                "unique_sessions": (
                    group["session_id"].map(non_empty_text).dropna().nunique()
                    if "session_id" in group
                    else None
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["event_count", "device_category"], ascending=[False, True]
    ).reset_index(drop=True)


def save_event_chart(distribution, output_dir):
    if distribution.empty:
        return
    figure, axis = plt.subplots(figsize=(10, 6))
    plot_data = distribution.sort_values("event_count")
    axis.barh(plot_data["event_name"], plot_data["event_count"])
    axis.set_title("User Event Distribution")
    axis.set_xlabel("Event count")
    axis.set_ylabel("Event name")
    figure.tight_layout()
    figure.savefig(output_dir / "event_distribution.png", dpi=150)
    plt.close(figure)


def pseudo_user_activity_summary(frame):
    """Summarize pseudo-user activity and its relationship with sessions."""
    empty_summary = {
        "total_pseudo_users": 0,
        "mean_events_per_pseudo_user": None,
        "median_events_per_pseudo_user": None,
        "q1_events_per_pseudo_user": None,
        "q3_events_per_pseudo_user": None,
        "p90_events_per_pseudo_user": None,
        "p95_events_per_pseudo_user": None,
        "p99_events_per_pseudo_user": None,
        "max_events_per_pseudo_user": None,
        "mean_sessions_per_pseudo_user": None,
        "median_sessions_per_pseudo_user": None,
        "pseudo_users_with_multiple_sessions": None,
        "pseudo_users_with_multiple_sessions_percentage": None,
    }
    if "user_psuedo_id" not in frame.columns:
        return pd.DataFrame([empty_summary]), pd.Series(dtype="int64")

    pseudo_ids = frame["user_psuedo_id"].map(non_empty_text)
    event_counts = pseudo_ids.dropna().value_counts()
    if event_counts.empty:
        return pd.DataFrame([empty_summary]), event_counts

    summary = {
        "total_pseudo_users": int(event_counts.size),
        "mean_events_per_pseudo_user": event_counts.mean(),
        "median_events_per_pseudo_user": event_counts.median(),
        "q1_events_per_pseudo_user": event_counts.quantile(0.25),
        "q3_events_per_pseudo_user": event_counts.quantile(0.75),
        "p90_events_per_pseudo_user": event_counts.quantile(0.90),
        "p95_events_per_pseudo_user": event_counts.quantile(0.95),
        "p99_events_per_pseudo_user": event_counts.quantile(0.99),
        "max_events_per_pseudo_user": event_counts.max(),
        "mean_sessions_per_pseudo_user": None,
        "median_sessions_per_pseudo_user": None,
        "pseudo_users_with_multiple_sessions": None,
        "pseudo_users_with_multiple_sessions_percentage": None,
    }

    if "session_id" in frame.columns:
        session_ids = frame["session_id"].map(non_empty_text)
        valid = pseudo_ids.notna() & session_ids.notna()
        sessions_per_pseudo = (
            pd.DataFrame(
                {"pseudo_id": pseudo_ids[valid], "session_id": session_ids[valid]}
            )
            .groupby("pseudo_id")["session_id"]
            .nunique()
        )
        if not sessions_per_pseudo.empty:
            multi_session_count = int((sessions_per_pseudo > 1).sum())
            summary.update(
                {
                    "mean_sessions_per_pseudo_user": sessions_per_pseudo.mean(),
                    "median_sessions_per_pseudo_user": sessions_per_pseudo.median(),
                    "pseudo_users_with_multiple_sessions": multi_session_count,
                    "pseudo_users_with_multiple_sessions_percentage": (
                        multi_session_count / event_counts.size * 100
                    ),
                }
            )
    return pd.DataFrame([summary]), event_counts


def save_pseudo_user_activity_chart(event_counts, output_dir):
    """Plot pseudo-user event counts through P99 without dropping source values."""
    if event_counts.empty:
        return
    p99 = event_counts.quantile(0.99)
    visible_counts = event_counts[event_counts <= p99]
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.hist(visible_counts, bins=30, edgecolor="white")
    axis.set_xlim(0, p99)
    axis.set_title("Distribution of Events per Pseudo User")
    axis.set_xlabel("Number of events per user_psuedo_id")
    axis.set_ylabel("Number of user_psuedo_id values")
    figure.tight_layout()
    figure.savefig(output_dir / "events_per_pseudo_user.png", dpi=150)
    plt.close(figure)


def main():
    events = load_events()
    distribution = event_distribution(events)
    output_dir = OUTPUT_DIR / "user_behavior" / "event_overview"
    output_dir.mkdir(parents=True, exist_ok=True)

    kpi_summary(events, distribution).to_csv(output_dir / "kpi_summary.csv", index=False)
    distribution.to_csv(output_dir / "event_distribution.csv", index=False)
    activity_summary(events, "user_id", "user").to_csv(
        output_dir / "user_activity_summary.csv", index=False
    )
    activity_summary(events, "session_id", "session").to_csv(
        output_dir / "session_activity_summary.csv", index=False
    )
    device = device_distribution(events)
    if device is not None:
        device.to_csv(output_dir / "device_distribution.csv", index=False)
    save_event_chart(distribution, output_dir)
    pseudo_summary, pseudo_event_counts = pseudo_user_activity_summary(events)
    pseudo_summary.to_csv(
        output_dir / "pseudo_user_activity_summary.csv", index=False
    )
    save_pseudo_user_activity_chart(pseudo_event_counts, output_dir)

    print(f"Saved: {output_dir}")
    print(f"Total events: {len(events)}")
    print(f"Event distribution total: {int(distribution['event_count'].sum())}")
    print(f"Event percentage total: {distribution['event_percentage'].sum():.6f}%")


if __name__ == "__main__":
    main()