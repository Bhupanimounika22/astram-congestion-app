import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
import folium
import h3
from streamlit_folium import st_folium

warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="ASTRAM | Event-Driven Congestion Intelligence",
    page_icon="🚦",
    layout="wide"
)

# ============================================================
# STYLES
# ============================================================
st.markdown("""
<style>
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}
.metric-card {
    background: #111827;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.08);
}
.small-note {
    font-size: 0.92rem;
    color: #bdbdbd;
}
.section-card {
    background: rgba(255,255,255,0.02);
    padding: 16px 18px;
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.08);
}
.example-card {
    background: rgba(255,255,255,0.03);
    padding: 12px;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 10px;
}
.example-title {
    font-size: 1rem;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# PATHS
# ============================================================
MODEL_DIR = "final_safe_model_artifacts"
HOTSPOT_PATH = "rebuild_outputs/h3_hotspot_summary.csv"
STEP1_PATH = "rebuild_outputs/step1_rebuilt_features.csv"

# ============================================================
# SAFE HELPERS
# ============================================================
def safe_text(x, default="unknown"):
    if pd.isna(x):
        return default
    x = str(x).strip()
    return x if x else default

def safe_float(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except:
        return default

def safe_int(x, default=0):
    try:
        if pd.isna(x):
            return default
        return int(float(x))
    except:
        return default

def to_arrow_safe(df):
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "object":
            out[c] = out[c].astype(str)
    return out

# ============================================================
# LOADERS
# ============================================================
@st.cache_data(show_spinner=False)
def load_step1_data():
    if os.path.exists(STEP1_PATH):
        return pd.read_csv(STEP1_PATH)
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_hotspot_data():
    if os.path.exists(HOTSPOT_PATH):
        df = pd.read_csv(HOTSPOT_PATH)

        required_cols = [
            "h3_cell", "hotspot_score", "hotspot_level", "hotspot_rank",
            "h3_event_count", "h3_avg_impact", "h3_avg_duration",
            "h3_closure_rate", "is_h3_hotspot"
        ]
        for c in required_cols:
            if c not in df.columns:
                if c in ["hotspot_score", "h3_avg_impact", "h3_avg_duration", "h3_closure_rate"]:
                    df[c] = 0.0
                elif c in ["hotspot_rank", "h3_event_count", "is_h3_hotspot"]:
                    df[c] = 0
                else:
                    df[c] = "unknown"

        if "cell_lat" not in df.columns or "cell_lng" not in df.columns:
            cell_lats = []
            cell_lngs = []
            for c in df["h3_cell"].astype(str):
                try:
                    lat, lng = h3.cell_to_latlng(c)
                except:
                    lat, lng = np.nan, np.nan
                cell_lats.append(lat)
                cell_lngs.append(lng)
            df["cell_lat"] = cell_lats
            df["cell_lng"] = cell_lngs

        return df
    return pd.DataFrame()

@st.cache_resource(show_spinner=False)
def load_models():
    models = {
        "closure_model": None,
        "closure_features": [],
        "closure_cat_cols": [],
        "duration_model": None,
        "duration_features": [],
        "duration_cat_cols": []
    }

    try:
        p = os.path.join(MODEL_DIR, "closure_classifier_catboost.pkl")
        if os.path.exists(p):
            models["closure_model"] = joblib.load(p)
    except Exception:
        models["closure_model"] = None

    try:
        p = os.path.join(MODEL_DIR, "closure_feature_columns.pkl")
        if os.path.exists(p):
            models["closure_features"] = joblib.load(p)
    except Exception:
        models["closure_features"] = []

    try:
        p = os.path.join(MODEL_DIR, "closure_categorical_feature_names.pkl")
        if os.path.exists(p):
            models["closure_cat_cols"] = joblib.load(p)
    except Exception:
        models["closure_cat_cols"] = []

    try:
        p = os.path.join(MODEL_DIR, "duration_regressor_catboost.pkl")
        if os.path.exists(p):
            models["duration_model"] = joblib.load(p)
    except Exception:
        models["duration_model"] = None

    try:
        p = os.path.join(MODEL_DIR, "duration_feature_columns.pkl")
        if os.path.exists(p):
            models["duration_features"] = joblib.load(p)
    except Exception:
        models["duration_features"] = []

    try:
        p = os.path.join(MODEL_DIR, "duration_categorical_feature_names.pkl")
        if os.path.exists(p):
            models["duration_cat_cols"] = joblib.load(p)
    except Exception:
        models["duration_cat_cols"] = []

    return models

base_df = load_step1_data()
hotspot_df = load_hotspot_data()
models = load_models()

# ============================================================
# VALUE LISTS FROM DATA
# ============================================================
def get_unique_values(df, col, fallback):
    if df.empty or col not in df.columns:
        return fallback
    vals = (
        df[col]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", np.nan)
        .dropna()
        .unique()
        .tolist()
    )
    vals = sorted(list(set(vals)))
    return vals if vals else fallback

event_type_opts = get_unique_values(base_df, "event_type", ["planned", "unplanned"])
event_cause_opts = get_unique_values(
    base_df, "event_cause",
    ["accident", "breakdown", "construction", "waterlogging", "treefall", "event", "procession", "other"]
)
priority_opts = get_unique_values(base_df, "priority", ["low", "medium", "high"])
corridor_opts = get_unique_values(base_df, "corridor", ["Outer Ring Road", "Bellary Road", "Bannerghatta Road", "unknown"])
junction_opts = get_unique_values(base_df, "junction", ["Silk Board", "Hebbal", "Marathahalli", "unknown"])
veh_type_opts = get_unique_values(base_df, "veh_type", ["car", "truck", "bus", "bike", "unknown"])
cargo_opts = get_unique_values(base_df, "cargo_material", ["unknown"])
reason_opts = get_unique_values(base_df, "reason_breakdown", ["unknown"])
route_opts = get_unique_values(base_df, "route_path", ["unknown"])

# ============================================================
# FINAL FE EXAMPLES (CALIBRATED)
# ============================================================
FE_EXAMPLES = {
    "Low": {
        "event_type": "planned",
        "event_cause": "construction",
        "priority": "low",
        "corridor": "Bannerghatta Road",
        "junction": "unknown",
        "veh_type": "car",
        "cargo_material": "unknown",
        "reason_breakdown": "unknown",
        "route_path": "unknown",
        "month_name": "June",
        "hour": 14,
        "address": "Near local service road",
        "latitude": 12.9177,
        "longitude": 77.6238,
        "description": "Minor planned maintenance on side lane. Traffic moving normally."
    },
    "Medium": {
        "event_type": "planned",
        "event_cause": "event",
        "priority": "medium",
        "corridor": "Bellary Road",
        "junction": "unknown",
        "veh_type": "car",
        "cargo_material": "unknown",
        "reason_breakdown": "unknown",
        "route_path": "unknown",
        "month_name": "June",
        "hour": 16,
        "address": "Near Bellary Road stretch",
        "latitude": 13.0200,
        "longitude": 77.5900,
        "description": "Public event causing moderate slowdown near the stretch."
    },
    "High": {
        "event_type": "unplanned",
        "event_cause": "accident",
        "priority": "high",
        "corridor": "Outer Ring Road",
        "junction": "Marathahalli",
        "veh_type": "truck",
        "cargo_material": "unknown",
        "reason_breakdown": "unknown",
        "route_path": "unknown",
        "month_name": "June",
        "hour": 18,
        "address": "Marathahalli ORR stretch",
        "latitude": 12.9591,
        "longitude": 77.6974,
        "description": "Accident involving truck causing lane blockage and traffic congestion."
    },
    "Critical": {
        "event_type": "unplanned",
        "event_cause": "accident",
        "priority": "high",
        "corridor": "Outer Ring Road",
        "junction": "Silk Board",
        "veh_type": "truck",
        "cargo_material": "unknown",
        "reason_breakdown": "unknown",
        "route_path": "unknown",
        "month_name": "June",
        "hour": 19,
        "address": "Silk Board Junction",
        "latitude": 12.9176,
        "longitude": 77.6235,
        "description": "Major truck accident with overturned vehicle, blocked carriageway and severe congestion."
    }
}

def get_best_option(value, options, default=None):
    if value in options:
        return value
    value_l = str(value).strip().lower()
    for opt in options:
        if str(opt).strip().lower() == value_l:
            return opt
    return default if default is not None else (options[0] if options else value)

# ============================================================
# FEATURE ENGINEERING
# ============================================================
def priority_to_num(p):
    p = safe_text(p).lower()
    if p == "high":
        return 3
    elif p == "medium":
        return 2
    return 1

def desc_flags(desc):
    d = safe_text(desc, "").lower()
    return {
        "desc_has_accident": int(any(k in d for k in ["accident", "collision", "crash"])),
        "desc_has_breakdown": int(any(k in d for k in ["breakdown", "stalled", "failed"])),
        "desc_has_truck": int("truck" in d or "lorry" in d),
        "desc_has_blocked": int(any(k in d for k in ["blocked", "obstruction", "blockage"])),
        "desc_has_jam": int(any(k in d for k in ["jam", "congestion", "traffic", "queue"])),
        "desc_has_overturned": int(any(k in d for k in ["overturned", "capsized"])),
        "desc_has_treefall": int(any(k in d for k in ["tree", "treefall"])),
        "desc_has_waterlogging": int(any(k in d for k in ["water", "waterlogging", "flood"])),
        "desc_has_construction": int(any(k in d for k in ["construction", "repair", "maintenance"])),
        "desc_has_event": int(any(k in d for k in ["event", "rally", "festival", "procession", "match"]))
    }

def build_base_feature_row(
    event_type, event_cause, priority, corridor, junction,
    veh_type, cargo_material, reason_breakdown, route_path,
    month, hour, address, latitude, longitude, description
):
    row = {}

    row["event_type"] = safe_text(event_type)
    row["event_cause"] = safe_text(event_cause)
    row["veh_type"] = safe_text(veh_type)
    row["priority"] = safe_text(priority)
    row["cargo_material"] = safe_text(cargo_material)
    row["reason_breakdown"] = safe_text(reason_breakdown)
    row["corridor"] = safe_text(corridor)
    row["junction"] = safe_text(junction)
    row["address"] = safe_text(address)
    row["route_path"] = safe_text(route_path)

    row["latitude"] = safe_float(latitude, 0.0)
    row["longitude"] = safe_float(longitude, 0.0)
    row["hour"] = safe_int(hour, 12)
    row["month"] = safe_int(month, 6)

    row["is_weekend"] = 0
    row["is_peak_hour"] = int(row["hour"] in [8, 9, 10, 17, 18, 19, 20])
    row["is_morning_peak"] = int(row["hour"] in [8, 9, 10])
    row["is_evening_peak"] = int(row["hour"] in [17, 18, 19, 20])
    row["is_night"] = int(row["hour"] >= 22 or row["hour"] <= 5)

    flags = desc_flags(description)
    row.update(flags)

    v = row["veh_type"].lower()
    row["is_heavy_vehicle"] = int(v in ["truck", "lorry", "bus", "tanker", "container"])
    row["is_commercial_vehicle"] = int(v in ["truck", "lorry", "bus", "tanker", "container", "tempo", "goods vehicle"])

    row["priority_num"] = priority_to_num(priority)
    row["is_planned_event"] = int(safe_text(event_type).lower() == "planned")
    row["is_unplanned_event"] = int(safe_text(event_type).lower() == "unplanned")

    row["has_start_location"] = int(row["latitude"] != 0 and row["longitude"] != 0)

    return row

# ============================================================
# HOTSPOT LOOKUP
# ============================================================
def get_hotspot_features(lat, lng):
    default = {
        "h3_cell": "unknown",
        "hotspot_score": 0.0,
        "hotspot_level": "Low",
        "hotspot_rank": 9999,
        "h3_event_count": 0,
        "h3_avg_impact": 0.0,
        "h3_avg_duration": 0.0,
        "h3_closure_rate": 0.0,
        "is_h3_hotspot": 0,
        "hotspot_source": "default"
    }

    if hotspot_df.empty:
        return default

    try:
        input_cell = h3.latlng_to_cell(float(lat), float(lng), 8)
    except:
        return default

    exact = hotspot_df[hotspot_df["h3_cell"].astype(str) == str(input_cell)]
    if not exact.empty:
        r = exact.iloc[0]
        return {
            "h3_cell": safe_text(r.get("h3_cell", input_cell)),
            "hotspot_score": safe_float(r.get("hotspot_score", 0.0)),
            "hotspot_level": safe_text(r.get("hotspot_level", "Low")).title(),
            "hotspot_rank": safe_int(r.get("hotspot_rank", 9999)),
            "h3_event_count": safe_int(r.get("h3_event_count", 0)),
            "h3_avg_impact": safe_float(r.get("h3_avg_impact", 0.0)),
            "h3_avg_duration": safe_float(r.get("h3_avg_duration", 0.0)),
            "h3_closure_rate": safe_float(r.get("h3_closure_rate", 0.0)),
            "is_h3_hotspot": safe_int(r.get("is_h3_hotspot", 0)),
            "hotspot_source": "exact_h3_match"
        }

    temp = hotspot_df.copy()
    temp = temp.dropna(subset=["cell_lat", "cell_lng"]).copy()
    if temp.empty:
        default["h3_cell"] = str(input_cell)
        return default

    temp["dist"] = np.sqrt(
        (temp["cell_lat"] - float(lat)) ** 2 +
        (temp["cell_lng"] - float(lng)) ** 2
    )
    nearest = temp.sort_values("dist").iloc[0]

    if safe_float(nearest["dist"], 999) > 0.03:
        default["h3_cell"] = str(input_cell)
        return default

    return {
        "h3_cell": safe_text(nearest.get("h3_cell", input_cell)),
        "hotspot_score": safe_float(nearest.get("hotspot_score", 0.0)),
        "hotspot_level": safe_text(nearest.get("hotspot_level", "Low")).title(),
        "hotspot_rank": safe_int(nearest.get("hotspot_rank", 9999)),
        "h3_event_count": safe_int(nearest.get("h3_event_count", 0)),
        "h3_avg_impact": safe_float(nearest.get("h3_avg_impact", 0.0)),
        "h3_avg_duration": safe_float(nearest.get("h3_avg_duration", 0.0)),
        "h3_closure_rate": safe_float(nearest.get("h3_closure_rate", 0.0)),
        "is_h3_hotspot": safe_int(nearest.get("is_h3_hotspot", 0)),
        "hotspot_source": "nearest_h3_fallback"
    }

# ============================================================
# AGG FEATURES FROM STEP-1
# ============================================================
def lookup_dataset_aggregates(row):
    defaults = {
        "corridor_event_count": 0,
        "junction_event_count": 0,
        "event_cause_event_count": 0,
        "corridor_cause_count": 0,
        "corridor_avg_impact": 0.0,
        "junction_avg_impact": 0.0
    }

    if base_df.empty:
        return defaults

    df = base_df.copy()

    corridor = safe_text(row.get("corridor", "unknown"))
    junction = safe_text(row.get("junction", "unknown"))
    cause = safe_text(row.get("event_cause", "unknown"))

    out = defaults.copy()

    if "corridor" in df.columns:
        cdf = df[df["corridor"].astype(str) == corridor]
        out["corridor_event_count"] = len(cdf)
        if len(cdf) > 0 and "impact_score_rule" in cdf.columns:
            out["corridor_avg_impact"] = safe_float(cdf["impact_score_rule"].mean(), 0.0)

    if "junction" in df.columns:
        jdf = df[df["junction"].astype(str) == junction]
        out["junction_event_count"] = len(jdf)
        if len(jdf) > 0 and "impact_score_rule" in jdf.columns:
            out["junction_avg_impact"] = safe_float(jdf["impact_score_rule"].mean(), 0.0)

    if "event_cause" in df.columns:
        edf = df[df["event_cause"].astype(str) == cause]
        out["event_cause_event_count"] = len(edf)

    if "corridor" in df.columns and "event_cause" in df.columns:
        ccdf = df[(df["corridor"].astype(str) == corridor) & (df["event_cause"].astype(str) == cause)]
        out["corridor_cause_count"] = len(ccdf)

    return out

# ============================================================
# FINAL CALIBRATED IMPACT SCORING
# ============================================================
def compute_rule_based_impact(row, hotspot):
    """
    Calibrated rule-based impact scoring so FE examples map correctly:
    Low -> Low
    Medium -> Medium
    High -> High
    Critical -> Critical
    """
    score = 0
    reasons = []

    event_type = safe_text(row["event_type"]).lower()
    cause = safe_text(row["event_cause"]).lower()
    priority = safe_text(row["priority"]).lower()
    corridor = safe_text(row["corridor"]).lower()
    junction = safe_text(row["junction"]).lower()

    # --------------------------------------------------------
    # 1) BASE BY EVENT CAUSE (reduced weights)
    # --------------------------------------------------------
    cause_map = {
        "breakdown": 10,
        "vehicle breakdown": 10,
        "accident": 22,
        "collision": 22,
        "construction": 8,
        "repair": 8,
        "maintenance": 8,
        "treefall": 16,
        "tree fall": 16,
        "waterlogging": 16,
        "flooding": 16,
        "event": 12,
        "festival": 12,
        "rally": 18,
        "procession": 16,
        "vip movement": 14,
        "protest": 18
    }

    base_cause = cause_map.get(cause, 8)
    score += base_cause
    reasons.append(f"Base score from event cause '{cause}' = {base_cause}")

    # --------------------------------------------------------
    # 2) PRIORITY (reduced)
    # --------------------------------------------------------
    if priority == "high":
        score += 8
        reasons.append("High operational priority increases impact.")
    elif priority == "medium":
        score += 4
        reasons.append("Medium priority adds moderate impact.")
    else:
        score += 1

    # --------------------------------------------------------
    # 3) PEAK HOUR (reduced)
    # --------------------------------------------------------
    if row["is_peak_hour"] == 1:
        score += 6
        reasons.append("Peak-hour traffic increases congestion sensitivity.")
    elif row["is_night"] == 1:
        score -= 2

    # --------------------------------------------------------
    # 4) EVENT TYPE (reduced)
    # --------------------------------------------------------
    if event_type == "planned":
        if cause in ["rally", "festival", "event", "procession", "vip movement", "protest"]:
            score += 3
            reasons.append("Planned public gathering may require traffic control.")
        else:
            score += 1
    else:
        score += 4
        reasons.append("Unplanned incident adds response urgency.")

    # --------------------------------------------------------
    # 5) DESCRIPTION FLAGS (reduced)
    # --------------------------------------------------------
    if row["desc_has_blocked"] == 1:
        score += 5
        reasons.append("Description indicates blocked lane / obstruction.")
    if row["desc_has_jam"] == 1:
        score += 4
        reasons.append("Description explicitly indicates traffic buildup.")
    if row["desc_has_overturned"] == 1:
        score += 8
        reasons.append("Overturned vehicle strongly increases disruption.")
    if row["desc_has_accident"] == 1 and cause != "accident":
        score += 3
    if row["desc_has_breakdown"] == 1 and cause != "breakdown":
        score += 2

    # --------------------------------------------------------
    # 6) HEAVY VEHICLE (reduced)
    # --------------------------------------------------------
    if row["is_heavy_vehicle"] == 1:
        score += 4
        reasons.append("Heavy/commercial vehicle may block more carriageway.")

    # --------------------------------------------------------
    # 7) CORRIDOR / JUNCTION BONUS (reduced)
    # --------------------------------------------------------
    corridor_bonus = 0
    critical_corridors = [
        "outer ring road", "orr", "bellary road", "hosur road",
        "old madras road", "bannerghatta road", "mysore road"
    ]
    if any(c in corridor for c in critical_corridors):
        corridor_bonus += 3

    if junction in ["silk board", "hebbal", "marathahalli", "tin factory", "kr puram"]:
        corridor_bonus += 2

    if corridor_bonus > 0:
        score += corridor_bonus
        reasons.append("Historically busy corridor / junction adds local traffic sensitivity.")

    # --------------------------------------------------------
    # 8) HOTSPOT BONUS (small only)
    # --------------------------------------------------------
    hs = safe_float(hotspot.get("hotspot_score", 0.0), 0.0)
    if hs >= 0.70:
        score += 3
        reasons.append("Location is inside a strong historical hotspot.")
    elif hs >= 0.45:
        score += 1
        reasons.append("Location has moderate hotspot history.")

    # --------------------------------------------------------
    # 9) FINAL SCORE + CALIBRATED BUCKETS
    # --------------------------------------------------------
    score = max(0, min(100, int(round(score))))

    # Calibrated thresholds
    # 0–18   = Low
    # 19–34  = Medium
    # 35–49  = High
    # 50+    = Critical
    if score <= 18:
        level = "Low"
    elif score <= 34:
        level = "Medium"
    elif score <= 49:
        level = "High"
    else:
        level = "Critical"

    return score, level, reasons

# ============================================================
# MODEL PREP
# ============================================================
def make_model_input(base_row, hotspot, aggs, feature_cols, cat_cols):
    row = {}
    merged = {}
    merged.update(base_row)
    merged.update(hotspot)
    merged.update(aggs)

    for c in feature_cols:
        row[c] = merged.get(c, np.nan)

    X = pd.DataFrame([row])

    for c in X.columns:
        if c in cat_cols:
            X[c] = X[c].fillna("unknown").astype(str)
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)

    return X

def predict_closure(base_row, hotspot, aggs):
    model = models["closure_model"]
    feat_cols = models["closure_features"]
    cat_cols = models["closure_cat_cols"]

    if model is None or len(feat_cols) == 0:
        return 0.25, "No"

    try:
        X = make_model_input(base_row, hotspot, aggs, feat_cols, cat_cols)

        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(X)[0][1])
        else:
            pred = model.predict(X)
            prob = float(pred[0]) if len(pred) > 0 else 0.25

        prob = max(0.0, min(1.0, prob))
        decision = "Yes" if prob >= 0.50 else "No"
        return prob, decision
    except Exception:
        return 0.25, "No"

def predict_duration(base_row, hotspot, aggs):
    model = models["duration_model"]
    feat_cols = models["duration_features"]
    cat_cols = models["duration_cat_cols"]

    if model is None or len(feat_cols) == 0:
        return 30

    try:
        X = make_model_input(base_row, hotspot, aggs, feat_cols, cat_cols)
        pred = model.predict(X)
        duration = float(pred[0]) if len(pred) > 0 else 30.0
        duration = max(5, min(720, duration))
        return int(round(duration))
    except Exception:
        return 30

# ============================================================
# FIXED PRACTICAL OPERATIONAL PLAN
# ============================================================
def get_operational_plan(impact_level, impact_score, closure_prob, closure_decision, duration_min, hotspot):
    hs = safe_float(hotspot.get("hotspot_score", 0.0), 0.0)

    if impact_level == "Low":
        plan = {
            "road_closure": "No closure required",
            "manpower": "2 traffic personnel",
            "barricades": "2 barricades, 4 cones",
            "diversion": "No diversion required. Keep one local bypass / service-road option ready if traffic builds.",
            "summary": "Low disruption expected. Local traffic monitoring is sufficient."
        }

    elif impact_level == "Medium":
        plan = {
            "road_closure": "No full closure. Use temporary lane control if congestion increases.",
            "manpower": "4 traffic personnel",
            "barricades": "4 barricades, 8 cones",
            "diversion": "Prepare a short local diversion around the affected stretch using nearby service roads / parallel streets.",
            "summary": "Moderate disruption expected. Moderate field deployment recommended."
        }

    elif impact_level == "High":
        plan = {
            "road_closure": "Partial closure / lane closure likely required",
            "manpower": "6–8 traffic personnel",
            "barricades": "6 barricades, 12 cones",
            "diversion": "Activate corridor-level diversion. Reroute vehicles to alternate arterial roads before the affected junction.",
            "summary": "High traffic disruption expected. Corridor-level intervention and active diversion planning recommended."
        }

    else:  # Critical
        plan = {
            "road_closure": "Strongly consider full closure or major diversion",
            "manpower": "10–14 traffic personnel + supervisor",
            "barricades": "10 barricades, 20+ cones",
            "diversion": "Activate major diversion plan immediately. Reroute traffic before hotspot entry and coordinate with police / control room for traffic regulation.",
            "summary": "Severe traffic breakdown risk. Immediate multi-team response recommended."
        }

    # Optional small adjustments without changing impact class
    if closure_decision == "Yes" or closure_prob >= 0.65:
        if impact_level == "Low":
            plan["road_closure"] = "No full closure, but prepare temporary lane restriction if obstruction remains."
        elif impact_level == "Medium":
            plan["road_closure"] = "Controlled lane closure may be required if traffic queues increase."
        elif impact_level == "High":
            plan["road_closure"] = "Partial road closure / controlled lane closure recommended."
        else:
            plan["road_closure"] = "Strongly consider full closure or major diversion"

    if duration_min >= 90:
        plan["summary"] += " Long disruption window expected; sustained deployment will be needed."
    elif duration_min <= 15:
        plan["summary"] += " Incident is expected to clear relatively quickly."

    if hs >= 0.60:
        plan["summary"] += " Location lies in a historically sensitive hotspot zone."

    return plan

# ============================================================
# MAP HELPERS
# ============================================================
def get_hotspot_color(score):
    if score >= 0.65:
        return "#d73027"   # red
    elif score >= 0.45:
        return "#fc8d59"   # orange
    elif score >= 0.25:
        return "#fee08b"   # yellow
    else:
        return "#91cf60"   # green

def draw_hotspot_map(lat, lng, hotspot, show_all_hotspots=True):
    m = folium.Map(location=[lat, lng], zoom_start=12, tiles="CartoDB positron")
    predicted_cell = str(hotspot.get("h3_cell", "unknown"))

    # --------------------------------------------------------
    # Draw ALL hotspot polygons
    # --------------------------------------------------------
    if not hotspot_df.empty:
        temp = hotspot_df.copy()
        temp = temp.dropna(subset=["h3_cell"])

        # if you want only nearby top cells, change show_all_hotspots=False
        if not show_all_hotspots:
            temp = temp.dropna(subset=["cell_lat", "cell_lng"]).copy()
            temp["dist"] = np.sqrt((temp["cell_lat"] - lat) ** 2 + (temp["cell_lng"] - lng) ** 2)
            temp = temp.sort_values("dist").head(20)

        for _, r in temp.iterrows():
            try:
                cell = str(r["h3_cell"])
                boundary = h3.cell_to_boundary(cell)
                poly = [(p[0], p[1]) for p in boundary]

                hs = safe_float(r.get("hotspot_score", 0.0))
                level = safe_text(r.get("hotspot_level", "low")).title()
                event_count = safe_int(r.get("h3_event_count", 0))
                avg_impact = safe_float(r.get("h3_avg_impact", 0.0))
                avg_duration = safe_float(r.get("h3_avg_duration", 0.0))

                is_predicted = (cell == predicted_cell)
                color = "#0033ff" if is_predicted else get_hotspot_color(hs)
                weight = 5 if is_predicted else 2
                fill_opacity = 0.40 if is_predicted else 0.18

                popup_txt = f"""
                <b>H3 Cell:</b> {cell}<br>
                <b>Hotspot Score:</b> {hs:.3f}<br>
                <b>Level:</b> {level}<br>
                <b>Events:</b> {event_count}<br>
                <b>Avg Impact:</b> {avg_impact:.1f}<br>
                <b>Avg Duration:</b> {avg_duration:.1f} min<br>
                <b>Predicted Event Cell:</b> {"YES" if is_predicted else "NO"}
                """

                folium.Polygon(
                    locations=poly,
                    color=color,
                    weight=weight,
                    fill=True,
                    fill_color=color,
                    fill_opacity=fill_opacity,
                    popup=popup_txt,
                    tooltip=("Predicted Hotspot Cell" if is_predicted else f"{level} hotspot")
                ).add_to(m)

                if is_predicted and "cell_lat" in r and "cell_lng" in r:
                    if not pd.isna(r["cell_lat"]) and not pd.isna(r["cell_lng"]):
                        folium.Marker(
                            [r["cell_lat"], r["cell_lng"]],
                            tooltip="Predicted Hotspot Cell",
                            popup=f"Predicted hotspot cell<br>{cell}",
                            icon=folium.Icon(color="blue", icon="star")
                        ).add_to(m)
            except:
                pass

    # Event point marker
    folium.Marker(
        [lat, lng],
        tooltip="Event Location",
        popup=f"Event Location<br>Lat: {lat:.5f}<br>Lng: {lng:.5f}",
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(m)

    return m

# ============================================================
# SESSION STATE DEFAULTS
# ============================================================
if "form_values" not in st.session_state:
    st.session_state.form_values = FE_EXAMPLES["Low"].copy()

def apply_example(example_name):
    ex = FE_EXAMPLES[example_name].copy()
    ex["event_type"] = get_best_option(ex["event_type"], event_type_opts, ex["event_type"])
    ex["event_cause"] = get_best_option(ex["event_cause"], event_cause_opts, ex["event_cause"])
    ex["priority"] = get_best_option(ex["priority"], priority_opts, ex["priority"])
    ex["corridor"] = get_best_option(ex["corridor"], corridor_opts, ex["corridor"])
    ex["junction"] = get_best_option(ex["junction"], junction_opts, ex["junction"])
    ex["veh_type"] = get_best_option(ex["veh_type"], veh_type_opts, ex["veh_type"])
    ex["cargo_material"] = get_best_option(ex["cargo_material"], cargo_opts, ex["cargo_material"])
    ex["reason_breakdown"] = get_best_option(ex["reason_breakdown"], reason_opts, ex["reason_breakdown"])
    ex["route_path"] = get_best_option(ex["route_path"], route_opts, ex["route_path"])
    st.session_state.form_values = ex

# ============================================================
# UI HEADER
# ============================================================
st.title("🚦 ASTRAM — Event-Driven Congestion Intelligence")
st.caption(
    "Hybrid system for Bengaluru traffic operations: "
    "**rule-based impact scoring + CatBoost closure/duration prediction + H3 hotspot intelligence + operational recommendations**"
)

# ============================================================
# EXAMPLE SECTION
# ============================================================
st.markdown("## 🎯 FE Example Scenarios")

ec1, ec2, ec3, ec4 = st.columns(4)

with ec1:
    st.markdown("""
    <div class="example-card">
        <div class="example-title">🟢 Low</div>
        Planned construction / minor maintenance / non-peak / no blockage
    </div>
    """, unsafe_allow_html=True)
    if st.button("Use Low Example", use_container_width=True):
        apply_example("Low")

with ec2:
    st.markdown("""
    <div class="example-card">
        <div class="example-title">🟡 Medium</div>
        Planned event / medium priority / moderate slowdown
    </div>
    """, unsafe_allow_html=True)
    if st.button("Use Medium Example", use_container_width=True):
        apply_example("Medium")

with ec3:
    st.markdown("""
    <div class="example-card">
        <div class="example-title">🟠 High</div>
        Unplanned accident / heavy vehicle / blocked lane / traffic congestion
    </div>
    """, unsafe_allow_html=True)
    if st.button("Use High Example", use_container_width=True):
        apply_example("High")

with ec4:
    st.markdown("""
    <div class="example-card">
        <div class="example-title">🔴 Critical</div>
        Major truck accident / overturned / severe blockage / hotspot corridor
    </div>
    """, unsafe_allow_html=True)
    if st.button("Use Critical Example", use_container_width=True):
        apply_example("Critical")

st.markdown("---")

# ============================================================
# FORM
# ============================================================
st.markdown("## 📝 Event Input")

fv = st.session_state.form_values

with st.form("predict_form", clear_on_submit=False):
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        event_type = st.selectbox(
            "Event Type", event_type_opts,
            index=event_type_opts.index(get_best_option(fv["event_type"], event_type_opts, event_type_opts[0]))
        )
        event_cause = st.selectbox(
            "Event Cause", event_cause_opts,
            index=event_cause_opts.index(get_best_option(fv["event_cause"], event_cause_opts, event_cause_opts[0]))
        )
        priority = st.selectbox(
            "Priority", priority_opts,
            index=priority_opts.index(get_best_option(fv["priority"], priority_opts, priority_opts[0]))
        )
        corridor = st.selectbox(
            "Corridor", corridor_opts,
            index=corridor_opts.index(get_best_option(fv["corridor"], corridor_opts, corridor_opts[0]))
        )

    with c2:
        junction = st.selectbox(
            "Junction", junction_opts,
            index=junction_opts.index(get_best_option(fv["junction"], junction_opts, junction_opts[0]))
        )
        veh_type = st.selectbox(
            "Vehicle Type", veh_type_opts,
            index=veh_type_opts.index(get_best_option(fv["veh_type"], veh_type_opts, veh_type_opts[0]))
        )
        cargo_material = st.selectbox(
            "Cargo Material", cargo_opts,
            index=cargo_opts.index(get_best_option(fv["cargo_material"], cargo_opts, cargo_opts[0]))
        )
        reason_breakdown = st.selectbox(
            "Reason Breakdown", reason_opts,
            index=reason_opts.index(get_best_option(fv["reason_breakdown"], reason_opts, reason_opts[0]))
        )

    with c3:
        route_path = st.selectbox(
            "Route Path", route_opts,
            index=route_opts.index(get_best_option(fv["route_path"], route_opts, route_opts[0]))
        )
        month_names = ["January","February","March","April","May","June","July","August","September","October","November","December"]
        month_name = st.selectbox(
            "Month",
            month_names,
            index=month_names.index(fv["month_name"]) if fv["month_name"] in month_names else 5
        )
        hour = st.slider("Hour of Day", 0, 23, int(fv["hour"]))
        address = st.text_input("Address / Landmark", value=fv["address"])

    with c4:
        latitude = st.number_input("Latitude", value=float(fv["latitude"]), format="%.6f")
        longitude = st.number_input("Longitude", value=float(fv["longitude"]), format="%.6f")
        description = st.text_area(
            "Event Description",
            value=fv["description"],
            height=110
        )

    st.markdown("""
    <div class="small-note">
    <b>Why these inputs?</b>  
    Event cause, priority, vehicle type, corridor, junction, time, and description are the strongest pre-event signals for estimating congestion impact.  
    Cargo / breakdown reason / route path are optional context fields used when available.
    </div>
    """, unsafe_allow_html=True)

    submitted = st.form_submit_button("Predict Event Impact")

# ============================================================
# PREDICTION
# ============================================================
if submitted:
    # persist latest values
    st.session_state.form_values = {
        "event_type": event_type,
        "event_cause": event_cause,
        "priority": priority,
        "corridor": corridor,
        "junction": junction,
        "veh_type": veh_type,
        "cargo_material": cargo_material,
        "reason_breakdown": reason_breakdown,
        "route_path": route_path,
        "month_name": month_name,
        "hour": hour,
        "address": address,
        "latitude": latitude,
        "longitude": longitude,
        "description": description
    }

    month_num = ["January","February","March","April","May","June","July","August","September","October","November","December"].index(month_name) + 1

    base_row = build_base_feature_row(
        event_type=event_type,
        event_cause=event_cause,
        priority=priority,
        corridor=corridor,
        junction=junction,
        veh_type=veh_type,
        cargo_material=cargo_material,
        reason_breakdown=reason_breakdown,
        route_path=route_path,
        month=month_num,
        hour=hour,
        address=address,
        latitude=latitude,
        longitude=longitude,
        description=description
    )

    hotspot = get_hotspot_features(latitude, longitude)
    aggs = lookup_dataset_aggregates(base_row)

    impact_score, impact_level, impact_reasons = compute_rule_based_impact(base_row, hotspot)
    closure_prob, closure_decision = predict_closure(base_row, hotspot, aggs)
    duration_min = predict_duration(base_row, hotspot, aggs)

    plan = get_operational_plan(
        impact_level=impact_level,
        impact_score=impact_score,
        closure_prob=closure_prob,
        closure_decision=closure_decision,
        duration_min=duration_min,
        hotspot=hotspot
    )

    result = {
        "impact_level": impact_level,
        "impact_score": impact_score,
        "closure_prob": closure_prob,
        "closure_decision": closure_decision,
        "duration_min": duration_min,
        "hotspot": hotspot,
        "impact_reasons": impact_reasons,
        "plan": plan
    }

    # ========================================================
    # HOTSPOT INTELLIGENCE
    # ========================================================
    st.markdown("---")
    st.markdown("## 🔥 Hotspot Intelligence")

    h1, h2, h3c, h4 = st.columns(4)
    with h1:
        st.metric("Hotspot Score", f"{result['hotspot']['hotspot_score']:.2f}")
    with h2:
        st.metric("Hotspot Level", result["hotspot"]["hotspot_level"])
    with h3c:
        st.metric("Predicted H3 Cell", result["hotspot"]["h3_cell"])
    with h4:
        rank_val = result["hotspot"]["hotspot_rank"]
        st.metric("Hotspot Rank", "N/A" if rank_val >= 9999 else rank_val)

    h5, h6, h7 = st.columns(3)
    with h5:
        st.metric("Historical Event Count", result["hotspot"]["h3_event_count"])
    with h6:
        st.metric("Avg Historical Impact", f"{result['hotspot']['h3_avg_impact']:.1f}")
    with h7:
        st.metric("Avg Historical Duration", f"{result['hotspot']['h3_avg_duration']:.1f} min")

    st.write(f"**Historical Closure Rate in this hotspot:** {result['hotspot']['h3_closure_rate']:.2f}")
    st.write(f"**Hotspot Lookup Mode:** {result['hotspot']['hotspot_source']}")

    if result["hotspot"]["hotspot_score"] >= 0.60:
        st.error("This location falls in a strong historical hotspot zone.")
    elif result["hotspot"]["hotspot_score"] >= 0.35:
        st.warning("This location has moderate hotspot history and may need proactive traffic control.")
    else:
        st.success("This location is not a major historical hotspot based on the H3 lookup.")

    # ========================================================
    # HOTSPOT MAP ABOVE PREDICTION
    # ========================================================
    st.markdown("## 🗺️ Hotspot Map")
    st.caption("All hotspot cells are displayed. The **predicted event hotspot cell** is highlighted in **blue**.")
    hotspot_map = draw_hotspot_map(latitude, longitude, result["hotspot"], show_all_hotspots=True)
    st_folium(hotspot_map, width=None, height=560, returned_objects=[])

    # ========================================================
    # PREDICTION SUMMARY
    # ========================================================
    st.markdown("## 📊 Prediction Summary")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Impact Level", result["impact_level"])
    with m2:
        st.metric("Impact Score", result["impact_score"])
    with m3:
        st.metric("Closure Probability", f"{result['closure_prob']*100:.1f}%")
    with m4:
        st.metric("Predicted Duration", f"{result['duration_min']} min")

    # ========================================================
    # RECOMMENDATION
    # ========================================================
    st.markdown("## 🚧 Operational Recommendation")
    r1, r2 = st.columns(2)

    with r1:
        st.success(f"**Road Closure Plan:** {result['plan']['road_closure']}")
        st.info(f"**Manpower Deployment:** {result['plan']['manpower']}")
        st.warning(f"**Barricading Requirement:** {result['plan']['barricades']}")

    with r2:
        st.info(f"**Diversion Route Plan:** {result['plan']['diversion']}")
        st.write(f"**Expected Duration Window:** {result['duration_min']} min")
        st.write(f"**Operational Summary:** {result['plan']['summary']}")

    # ========================================================
    # WHY THIS PREDICTION
    # ========================================================
    st.markdown("## 🧾 Why this prediction?")
    w1, w2, w3 = st.columns(3)

    with w1:
        st.markdown(f"### Impact = {result['impact_level']}")
        for r in result["impact_reasons"][:8]:
            st.markdown(f"- {r}")

    with w2:
        st.markdown(f"### Closure = {result['closure_decision']}")
        st.markdown(f"- Model road-closure probability: **{result['closure_prob']*100:.1f}%**.")
        if result["closure_prob"] >= 0.65:
            st.markdown("- Closure probability is high enough to prepare lane control / closure operations.")
        elif result["closure_prob"] >= 0.40:
            st.markdown("- Moderate closure probability; keep barricades and diversion staff ready.")
        else:
            st.markdown("- Closure probability is currently low; monitor before escalating.")

    with w3:
        st.markdown("### Duration")
        st.markdown(f"- Predicted disruption duration is approximately **{result['duration_min']} minutes**.")
        if result["duration_min"] >= 90:
            st.markdown("- Long incident window expected; sustained deployment may be needed.")
        elif result["duration_min"] <= 15:
            st.markdown("- Likely short-duration disruption if cleared quickly.")
        else:
            st.markdown("- Moderate disruption window expected.")

    # ========================================================
    # FINAL ACTION VIEW
    # ========================================================
    st.markdown("## 🧠 Final Action View")
    if result["impact_level"] == "Low":
        st.info("Low impact event detected. Local monitoring and minimal deployment recommended.")
    elif result["impact_level"] == "Medium":
        st.info("Medium impact event detected. Moderate deployment recommended.")
    elif result["impact_level"] == "High":
        st.warning("High impact event detected. Strong field response and corridor diversion planning recommended.")
    else:
        st.error("Critical impact event detected. Immediate multi-team response recommended.")

    # ========================================================
    # DEBUG / FEATURE SNAPSHOT
    # ========================================================
    with st.expander("Show model input snapshot"):
        debug_df = pd.DataFrame([{
            **base_row,
            **{
                "hotspot_score": hotspot["hotspot_score"],
                "hotspot_level": hotspot["hotspot_level"],
                "h3_cell": hotspot["h3_cell"],
                "h3_event_count": hotspot["h3_event_count"],
                "h3_avg_impact": hotspot["h3_avg_impact"],
                "h3_avg_duration": hotspot["h3_avg_duration"],
                "corridor_event_count": aggs["corridor_event_count"],
                "junction_event_count": aggs["junction_event_count"],
                "event_cause_event_count": aggs["event_cause_event_count"],
                "corridor_cause_count": aggs["corridor_cause_count"]
            }
        }])
        st.dataframe(to_arrow_safe(debug_df), use_container_width=True)

else:
    st.info("Choose an FE example or fill the event details manually, then click **Predict Event Impact**.")