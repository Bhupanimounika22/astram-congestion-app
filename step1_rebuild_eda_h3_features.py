import os
import re
import numpy as np
import pandas as pd
import h3
import folium

# ============================================================
# CONFIG
# ============================================================
INPUT_CSV = "Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"
OUTPUT_DIR = "rebuild_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

H3_RESOLUTION = 8

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(INPUT_CSV)

print("=" * 80)
print("RAW DATASET LOADED")
print("=" * 80)
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# ============================================================
# BASIC CLEANING
# ============================================================
def safe_text(x):
    if pd.isna(x):
        return "unknown"
    x = str(x).strip().lower()
    return x if x else "unknown"

text_cols = [
    "event_type", "event_cause", "requires_road_closure", "status", "direction",
    "description", "veh_type", "corridor", "priority", "cargo_material",
    "reason_breakdown", "route_path", "police_station", "resolved_at_address",
    "zone", "junction", "address", "end_address"
]

for col in text_cols:
    if col in df.columns:
        df[col] = df[col].apply(safe_text)

# ============================================================
# DATETIME PARSING
# ============================================================
for col in ["start_datetime", "end_datetime", "closed_datetime", "resolved_datetime", "modified_datetime", "created_date"]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

# use created_date as fallback if start_datetime missing
if "start_datetime" not in df.columns:
    df["start_datetime"] = pd.NaT

if "created_date" in df.columns:
    df["start_datetime"] = df["start_datetime"].fillna(df["created_date"])

# ============================================================
# DURATION
# ============================================================
# Prefer end_datetime, else resolved_datetime, else closed_datetime
end_proxy = None
for c in ["end_datetime", "resolved_datetime", "closed_datetime"]:
    if c in df.columns:
        if end_proxy is None:
            end_proxy = df[c].copy()
        else:
            end_proxy = end_proxy.fillna(df[c])

if end_proxy is None:
    df["duration_minutes"] = np.nan
else:
    df["duration_minutes"] = (end_proxy - df["start_datetime"]).dt.total_seconds() / 60.0

# keep only positive duration
df.loc[df["duration_minutes"] <= 0, "duration_minutes"] = np.nan

print("\n" + "=" * 80)
print("DURATION SUMMARY (RAW)")
print("=" * 80)
print(df["duration_minutes"].describe())

# ============================================================
# NUMERIC GEO CLEANING
# ============================================================
geo_cols = ["latitude", "longitude", "endlatitude", "endlongitude", "resolved_at_latitude", "resolved_at_longitude"]
for c in geo_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# use start location if available, else resolved location
df["use_lat"] = df["latitude"]
df["use_lng"] = df["longitude"]

if "resolved_at_latitude" in df.columns:
    df["use_lat"] = df["use_lat"].fillna(df["resolved_at_latitude"])
if "resolved_at_longitude" in df.columns:
    df["use_lng"] = df["use_lng"].fillna(df["resolved_at_longitude"])

df["has_start_location"] = (~df["latitude"].isna() & ~df["longitude"].isna()).astype(int)
df["has_resolved_location"] = (~df.get("resolved_at_latitude", pd.Series(index=df.index)).isna() &
                               ~df.get("resolved_at_longitude", pd.Series(index=df.index)).isna()).astype(int)

# ============================================================
# TIME FEATURES
# ============================================================
df["hour"] = df["start_datetime"].dt.hour.fillna(12).astype(int)
df["day_of_week"] = df["start_datetime"].dt.day_name().fillna("Unknown")
df["month"] = df["start_datetime"].dt.month.fillna(1).astype(int)

df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)
df["is_peak_hour"] = df["hour"].isin([8, 9, 10, 17, 18, 19, 20]).astype(int)
df["is_morning_peak"] = df["hour"].isin([8, 9, 10]).astype(int)
df["is_evening_peak"] = df["hour"].isin([17, 18, 19, 20]).astype(int)
df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)

# ============================================================
# DESCRIPTION FLAGS
# ============================================================
desc = df["description"].fillna("unknown").astype(str).str.lower()

def has_any(text_series, keywords):
    pattern = "|".join([re.escape(k) for k in keywords])
    return text_series.str.contains(pattern, regex=True, na=False).astype(int)

df["desc_has_accident"] = has_any(desc, ["accident", "collision", "crash"])
df["desc_has_breakdown"] = has_any(desc, ["breakdown", "stalled"])
df["desc_has_truck"] = has_any(desc, ["truck", "lorry"])
df["desc_has_blocked"] = has_any(desc, ["blocked", "blockage", "obstruction", "road blocked", "lane blocked"])
df["desc_has_jam"] = has_any(desc, ["jam", "congestion", "traffic jam"])
df["desc_has_overturned"] = has_any(desc, ["overturned", "pileup"])
df["desc_has_treefall"] = has_any(desc, ["tree", "treefall", "tree fall"])
df["desc_has_waterlogging"] = has_any(desc, ["waterlogging", "flood", "water logged"])
df["desc_has_construction"] = has_any(desc, ["construction", "repair", "road work"])
df["desc_has_event"] = has_any(desc, ["rally", "festival", "procession", "event", "gathering"])

# ============================================================
# VEHICLE / EVENT FLAGS
# ============================================================
HEAVY_VEHICLES = {"truck", "lorry", "trailer", "heavy vehicle", "bus"}

df["veh_type"] = df["veh_type"].fillna("unknown").astype(str).str.lower()
df["is_heavy_vehicle"] = df["veh_type"].isin(HEAVY_VEHICLES).astype(int)
df["is_commercial_vehicle"] = df["veh_type"].isin({"truck", "lorry", "tempo", "bus", "trailer", "heavy vehicle"}).astype(int)

df["event_type"] = df["event_type"].fillna("unknown").astype(str).str.lower()
df["is_planned_event"] = (df["event_type"] == "planned").astype(int)
df["is_unplanned_event"] = (df["event_type"] == "unplanned").astype(int)

priority_map = {"low": 1, "medium": 2, "high": 3}
df["priority_num"] = df["priority"].map(priority_map).fillna(1).astype(int)

# ============================================================
# ROAD CLOSURE FLAG
# ============================================================
closure_series = df["requires_road_closure"].fillna("unknown").astype(str).str.lower()

def closure_to_flag(x):
    if x in ["yes", "y", "true", "1", "required"]:
        return 1
    return 0

df["requires_road_closure_flag"] = closure_series.apply(closure_to_flag).astype(int)

# ============================================================
# RULE-BASED IMPACT SCORE
# ============================================================
def compute_rule_impact(row):
    score = 0
    cause = row.get("event_cause", "unknown")
    priority = row.get("priority", "low")

    if any(k in cause for k in ["accident", "collision", "crash"]):
        score += 28
    elif any(k in cause for k in ["rally", "festival", "procession", "event", "gathering"]):
        score += 30
    elif any(k in cause for k in ["construction", "repair"]):
        score += 20
    elif any(k in cause for k in ["tree", "waterlogging", "flood"]):
        score += 18
    elif any(k in cause for k in ["breakdown", "stalled"]):
        score += 14
    else:
        score += 8

    if priority == "high":
        score += 18
    elif priority == "medium":
        score += 10
    else:
        score += 4

    if row["is_peak_hour"] == 1:
        score += 10
    if row["desc_has_blocked"] == 1:
        score += 8
    if row["desc_has_jam"] == 1:
        score += 8
    if row["desc_has_overturned"] == 1:
        score += 12
    if row["is_heavy_vehicle"] == 1:
        score += 8

    # duration proxy if known
    dur = row.get("duration_minutes", np.nan)
    if pd.notna(dur):
        if dur >= 180:
            score += 10
        elif dur >= 60:
            score += 6
        elif dur >= 20:
            score += 3

    return score

df["impact_score_rule"] = df.apply(compute_rule_impact, axis=1)

def impact_label(score):
    if score <= 22:
        return "low"
    elif score <= 42:
        return "medium"
    elif score <= 70:
        return "high"
    return "critical"

df["impact_level"] = df["impact_score_rule"].apply(impact_label)

print("\n" + "=" * 80)
print("RULE IMPACT DISTRIBUTION")
print("=" * 80)
print(df["impact_level"].value_counts(dropna=False))

# ============================================================
# H3 CELL CREATION
# ============================================================
def latlng_to_h3(lat, lng, resolution=8):
    try:
        if pd.isna(lat) or pd.isna(lng):
            return np.nan
        return h3.latlng_to_cell(float(lat), float(lng), resolution)
    except Exception:
        return np.nan

df["h3_cell"] = df.apply(lambda r: latlng_to_h3(r["use_lat"], r["use_lng"], H3_RESOLUTION), axis=1)

valid_geo_df = df[~df["h3_cell"].isna()].copy()

print("\n" + "=" * 80)
print(f"ROWS WITH VALID GEO FOR H3: {len(valid_geo_df)}")
print("=" * 80)

# ============================================================
# CORRECT HOTSPOT SUMMARY
# ============================================================
# Use only rows with valid h3_cell
h3_summary = (
    valid_geo_df.groupby("h3_cell", dropna=False)
    .agg(
        h3_event_count=("h3_cell", "size"),
        h3_avg_impact=("impact_score_rule", "mean"),
        h3_avg_duration=("duration_minutes", "mean"),
        h3_closure_rate=("requires_road_closure_flag", "mean")
    )
    .reset_index()
)

h3_summary["h3_event_count"] = h3_summary["h3_event_count"].fillna(0).astype(int)
h3_summary["h3_avg_impact"] = h3_summary["h3_avg_impact"].fillna(0.0)
h3_summary["h3_avg_duration"] = h3_summary["h3_avg_duration"].fillna(0.0)
h3_summary["h3_closure_rate"] = h3_summary["h3_closure_rate"].fillna(0.0)

def safe_norm(s):
    s = s.fillna(0)
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mn) / (mx - mn)

h3_summary["event_count_norm"] = safe_norm(h3_summary["h3_event_count"])
h3_summary["impact_norm"] = safe_norm(h3_summary["h3_avg_impact"])
h3_summary["duration_norm"] = safe_norm(h3_summary["h3_avg_duration"])
h3_summary["closure_norm"] = safe_norm(h3_summary["h3_closure_rate"])

# final hotspot score
h3_summary["hotspot_score"] = (
    0.35 * h3_summary["event_count_norm"] +
    0.35 * h3_summary["impact_norm"] +
    0.20 * h3_summary["duration_norm"] +
    0.10 * h3_summary["closure_norm"]
)

h3_summary["hotspot_rank"] = h3_summary["hotspot_score"].rank(
    ascending=False, method="dense"
).astype(int)

def hotspot_level(score):
    if score >= 0.75:
        return "critical"
    elif score >= 0.50:
        return "high"
    elif score >= 0.25:
        return "medium"
    return "low"

h3_summary["hotspot_level"] = h3_summary["hotspot_score"].apply(hotspot_level)
h3_summary["is_h3_hotspot"] = (h3_summary["hotspot_score"] >= 0.50).astype(int)

# keep final columns only
h3_summary_final = h3_summary[
    [
        "h3_cell",
        "h3_event_count",
        "h3_avg_impact",
        "h3_avg_duration",
        "h3_closure_rate",
        "hotspot_score",
        "hotspot_level",
        "hotspot_rank",
        "is_h3_hotspot"
    ]
].copy()

h3_summary_final.to_csv(os.path.join(OUTPUT_DIR, "h3_hotspot_summary.csv"), index=False)
print(f"\nSaved corrected H3 hotspot summary: {os.path.join(OUTPUT_DIR, 'h3_hotspot_summary.csv')}")

# ============================================================
# MERGE HOTSPOT FEATURES BACK TO MAIN DATA
# ============================================================
df = df.merge(h3_summary_final, on="h3_cell", how="left")

for c in ["hotspot_score", "h3_avg_impact", "h3_avg_duration", "h3_closure_rate"]:
    df[c] = df[c].fillna(0.0)

for c in ["h3_event_count", "hotspot_rank", "is_h3_hotspot"]:
    df[c] = df[c].fillna(0)

df["hotspot_level"] = df["hotspot_level"].fillna("low")

# ============================================================
# SIMPLE HISTORICAL AGGREGATES FOR MODELS
# ============================================================
def group_stats(data, key_col, prefix):
    g = (
        data.groupby(key_col)
        .agg(
            event_count=(key_col, "size"),
            avg_impact=("impact_score_rule", "mean")
        )
        .reset_index()
    )
    g.columns = [key_col, f"{prefix}_event_count", f"{prefix}_avg_impact"]
    return g

corridor_stats = group_stats(df, "corridor", "corridor")
junction_stats = group_stats(df, "junction", "junction")
cause_stats = group_stats(df, "event_cause", "event_cause")

corridor_cause = (
    df.groupby(["corridor", "event_cause"])
    .size()
    .reset_index(name="corridor_cause_count")
)

df = df.merge(corridor_stats, on="corridor", how="left")
df = df.merge(junction_stats, on="junction", how="left")
df = df.merge(cause_stats, on="event_cause", how="left")
df = df.merge(corridor_cause, on=["corridor", "event_cause"], how="left")

for c in [
    "corridor_event_count", "corridor_avg_impact",
    "junction_event_count", "junction_avg_impact",
    "event_cause_event_count", "event_cause_avg_impact",
    "corridor_cause_count"
]:
    if c in df.columns:
        df[c] = df[c].fillna(0)

# ============================================================
# SAVE REBUILT FEATURE DATASET
# ============================================================
feature_cols_to_keep = [
    "event_type", "event_cause", "requires_road_closure", "status", "direction",
    "description", "veh_type", "corridor", "priority", "cargo_material",
    "reason_breakdown", "route_path", "police_station", "resolved_at_address",
    "zone", "junction", "address", "end_address",
    "latitude", "longitude", "endlatitude", "endlongitude",
    "resolved_at_latitude", "resolved_at_longitude",
    "use_lat", "use_lng",
    "duration_minutes",
    "hour", "day_of_week", "month",
    "is_weekend", "is_peak_hour", "is_morning_peak", "is_evening_peak", "is_night",
    "desc_has_accident", "desc_has_breakdown", "desc_has_truck", "desc_has_blocked",
    "desc_has_jam", "desc_has_overturned", "desc_has_treefall", "desc_has_waterlogging",
    "desc_has_construction", "desc_has_event",
    "is_heavy_vehicle", "is_commercial_vehicle", "priority_num",
    "is_planned_event", "is_unplanned_event",
    "impact_score_rule", "impact_level", "requires_road_closure_flag",
    "h3_cell", "hotspot_score", "hotspot_level", "hotspot_rank",
    "h3_event_count", "h3_avg_duration", "h3_avg_impact", "h3_closure_rate", "is_h3_hotspot",
    "corridor_event_count", "junction_event_count", "event_cause_event_count",
    "corridor_cause_count", "corridor_avg_impact", "junction_avg_impact",
    "has_start_location", "has_resolved_location"
]

feature_cols_to_keep = [c for c in feature_cols_to_keep if c in df.columns]
final_df = df[feature_cols_to_keep].copy()

final_df.to_csv(os.path.join(OUTPUT_DIR, "step1_rebuilt_features.csv"), index=False)

print("\n" + "=" * 80)
print("STEP-1 REBUILT FEATURE DATASET SAVED")
print("=" * 80)
print("Saved:", os.path.join(OUTPUT_DIR, "step1_rebuilt_features.csv"))
print("Shape:", final_df.shape)

# ============================================================
# MISSING / DISTRIBUTION REPORTS
# ============================================================
missing_summary = (df.isna().mean() * 100).sort_values(ascending=False).round(2)
missing_summary.to_csv(os.path.join(OUTPUT_DIR, "missing_value_summary.csv"))

df["impact_level"].value_counts(dropna=False).to_csv(os.path.join(OUTPUT_DIR, "impact_distribution.csv"))
df["requires_road_closure_flag"].value_counts(dropna=False).to_csv(os.path.join(OUTPUT_DIR, "closure_distribution.csv"))

# ============================================================
# HOTSPOT HTML MAP
# ============================================================
def hotspot_color(level):
    if level == "critical":
        return "#d73027"
    elif level == "high":
        return "#fc8d59"
    elif level == "medium":
        return "#fee08b"
    return "#91cf60"

if len(valid_geo_df) > 0:
    center_lat = valid_geo_df["use_lat"].median()
    center_lng = valid_geo_df["use_lng"].median()
else:
    center_lat, center_lng = 12.9716, 77.5946

m = folium.Map(location=[center_lat, center_lng], zoom_start=11, tiles="CartoDB positron")

map_df = h3_summary_final.sort_values("hotspot_score", ascending=False).head(80)

for _, r in map_df.iterrows():
    cell = r["h3_cell"]
    try:
        boundary = h3.cell_to_boundary(cell)
        polygon = [(lat, lng) for lat, lng in boundary]
        color = hotspot_color(r["hotspot_level"])

        popup_html = f"""
        <b>H3 Cell:</b> {cell}<br>
        <b>Hotspot Score:</b> {round(float(r['hotspot_score']), 3)}<br>
        <b>Hotspot Level:</b> {r['hotspot_level'].title()}<br>
        <b>Event Count:</b> {int(r['h3_event_count'])}<br>
        <b>Avg Impact:</b> {round(float(r['h3_avg_impact']), 2)}<br>
        <b>Avg Duration:</b> {round(float(r['h3_avg_duration']), 2)} min<br>
        <b>Closure Rate:</b> {round(float(r['h3_closure_rate']), 2)}
        """

        folium.Polygon(
            locations=polygon,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.35,
            weight=1,
            popup=folium.Popup(popup_html, max_width=260)
        ).add_to(m)
    except Exception:
        continue

hotspot_map_path = os.path.join(OUTPUT_DIR, "bangalore_hotspots.html")
m.save(hotspot_map_path)

print("\n" + "=" * 80)
print("HOTSPOT HTML MAP SAVED")
print("=" * 80)
print("Saved:", hotspot_map_path)

print("\n" + "=" * 80)
print("ALL STEP-1 OUTPUTS SAVED")
print("=" * 80)
print("Folder:", OUTPUT_DIR)
print("\nFiles created:")
print("- step1_rebuilt_features.csv")
print("- h3_hotspot_summary.csv")
print("- bangalore_hotspots.html")
print("- missing_value_summary.csv")
print("- impact_distribution.csv")
print("- closure_distribution.csv")