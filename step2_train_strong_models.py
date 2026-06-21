import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from catboost import CatBoostClassifier, CatBoostRegressor

# ============================================================
# CONFIG
# ============================================================
INPUT_FILE = "rebuild_outputs/step1_rebuilt_features.csv"
MODEL_DIR = "final_safe_model_artifacts"
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42
MIN_DURATION = 1
MAX_DURATION = 1440   # 24 hours cap

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(INPUT_FILE)

print("=" * 100)
print("STEP-2 FINAL SAFE MODEL TRAINING")
print("=" * 100)
print("Loaded rebuilt Step-1 data:", df.shape)
print("Columns:", df.columns.tolist())

# ============================================================
# HELPERS
# ============================================================
def safe_text(x):
    if pd.isna(x):
        return "unknown"
    x = str(x).strip().lower()
    return x if x else "unknown"

def fill_object_cols(frame):
    for col in frame.select_dtypes(include="object").columns:
        frame[col] = frame[col].fillna("unknown").astype(str)
    return frame

def fill_numeric_cols(frame):
    for col in frame.select_dtypes(include=[np.number]).columns:
        med = frame[col].median()
        if pd.isna(med):
            med = 0
        frame[col] = frame[col].fillna(med)
    return frame

# standardize text columns
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].apply(safe_text)

# ============================================================
# 1) DEFINE SAFE PRE-EVENT FEATURE SET
# ============================================================
# These are the columns we are willing to use for REAL prediction.
# Rule: should be reasonably available when an event is first reported.
#
# We intentionally remove:
# - endlatitude/endlongitude
# - resolved_at_* fields
# - end_address
# - status / zone / police_station
# - closure-derived historical leakage
# - duration-derived leakage-ish features for closure
#
# We keep:
# - event metadata
# - coarse geo (latitude/longitude of start)
# - time features
# - description flags
# - hotspot summary features that do NOT directly encode closure target
# - corridor/junction historical impact context

SAFE_PRE_EVENT_FEATURES = [
    # core event fields
    "event_type",
    "event_cause",
    "veh_type",
    "priority",
    "cargo_material",
    "reason_breakdown",
    "corridor",
    "junction",
    "address",

    # starting location only
    "latitude",
    "longitude",

    # time
    "hour",
    "month",
    "is_weekend",
    "is_peak_hour",
    "is_morning_peak",
    "is_evening_peak",
    "is_night",

    # text-derived flags
    "desc_has_accident",
    "desc_has_breakdown",
    "desc_has_truck",
    "desc_has_blocked",
    "desc_has_jam",
    "desc_has_overturned",
    "desc_has_treefall",
    "desc_has_waterlogging",
    "desc_has_construction",
    "desc_has_event",

    # vehicle / event flags
    "is_heavy_vehicle",
    "is_commercial_vehicle",
    "priority_num",
    "is_planned_event",
    "is_unplanned_event",

    # H3 / hotspot features that do NOT directly use closure target
    "h3_cell",
    "hotspot_score",
    "hotspot_level",
    "hotspot_rank",
    "h3_event_count",
    "h3_avg_impact",
    "is_h3_hotspot",

    # historical count / impact features
    "corridor_event_count",
    "junction_event_count",
    "event_cause_event_count",
    "corridor_cause_count",
    "corridor_avg_impact",
    "junction_avg_impact",

    # completeness
    "has_start_location"
]

SAFE_PRE_EVENT_FEATURES = [c for c in SAFE_PRE_EVENT_FEATURES if c in df.columns]

print("\n" + "=" * 100)
print("SAFE PRE-EVENT FEATURES")
print("=" * 100)
print("Number of safe features:", len(SAFE_PRE_EVENT_FEATURES))
print(SAFE_PRE_EVENT_FEATURES)

# ============================================================
# 2) TRAIN CLOSURE MODEL (SAFE VERSION)
# ============================================================
print("\n" + "=" * 100)
print("TRAINING SAFE CLOSURE MODEL")
print("=" * 100)

if "requires_road_closure_flag" not in df.columns:
    raise ValueError("requires_road_closure_flag not found in dataset")

closure_df = df.copy()

X_closure = closure_df[SAFE_PRE_EVENT_FEATURES].copy()
y_closure = closure_df["requires_road_closure_flag"].astype(int)

X_closure = fill_object_cols(X_closure)
X_closure = fill_numeric_cols(X_closure)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_closure,
    y_closure,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_closure
)

closure_feature_cols = X_train_c.columns.tolist()
closure_cat_cols = [c for c in closure_feature_cols if X_train_c[c].dtype == "object"]
closure_cat_idx = [closure_feature_cols.index(c) for c in closure_cat_cols]

pos = y_train_c.sum()
neg = len(y_train_c) - pos
scale_pos_weight = max(1.0, neg / max(pos, 1))

print("Closure class distribution:")
print(y_closure.value_counts())
print("Closure scale_pos_weight:", round(scale_pos_weight, 2))

closure_model = CatBoostClassifier(
    iterations=400,
    depth=5,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="F1",
    random_seed=RANDOM_STATE,
    verbose=100,
    scale_pos_weight=scale_pos_weight,
    l2_leaf_reg=8
)

closure_model.fit(
    X_train_c,
    y_train_c,
    cat_features=closure_cat_idx,
    eval_set=(X_test_c, y_test_c),
    use_best_model=True
)

pred_c = closure_model.predict(X_test_c).astype(int).flatten()
proba_c = closure_model.predict_proba(X_test_c)[:, 1]

print("\nClosure Classification Report:")
print(classification_report(y_test_c, pred_c))
print("Closure Accuracy :", accuracy_score(y_test_c, pred_c))
print("Closure Macro F1 :", f1_score(y_test_c, pred_c, average="macro"))
print("Closure Precision:", precision_score(y_test_c, pred_c, zero_division=0))
print("Closure Recall   :", recall_score(y_test_c, pred_c, zero_division=0))
print("\nClosure Confusion Matrix:")
print(confusion_matrix(y_test_c, pred_c))

closure_importance = pd.DataFrame({
    "feature": closure_feature_cols,
    "importance": closure_model.get_feature_importance()
}).sort_values("importance", ascending=False)

print("\nTop 25 Closure Features:")
print(closure_importance.head(25))

joblib.dump(closure_model, os.path.join(MODEL_DIR, "closure_classifier_catboost.pkl"))
joblib.dump(closure_feature_cols, os.path.join(MODEL_DIR, "closure_feature_columns.pkl"))
joblib.dump(closure_cat_cols, os.path.join(MODEL_DIR, "closure_categorical_feature_names.pkl"))
closure_importance.to_csv(os.path.join(MODEL_DIR, "closure_feature_importance.csv"), index=False)

# ============================================================
# 3) TRAIN DURATION MODEL (SAFE VERSION)
# ============================================================
print("\n" + "=" * 100)
print("TRAINING SAFE DURATION MODEL")
print("=" * 100)

if "duration_minutes" not in df.columns:
    raise ValueError("duration_minutes not found in dataset")

duration_df = df.copy()
duration_df["duration_minutes"] = pd.to_numeric(duration_df["duration_minutes"], errors="coerce")

# keep only realistic durations
duration_df = duration_df[
    duration_df["duration_minutes"].notna() &
    (duration_df["duration_minutes"] >= MIN_DURATION) &
    (duration_df["duration_minutes"] <= MAX_DURATION)
].copy()

print("Rows after duration cleaning:", duration_df.shape)
print("Duration distribution after cleaning:")
print(duration_df["duration_minutes"].describe())

if len(duration_df) < 200:
    print("Not enough clean rows to train duration model.")
else:
    # duration can use the same safe feature set
    # plus a few extra safe context features if present
    DURATION_FEATURES = SAFE_PRE_EVENT_FEATURES.copy()

    # optionally add non-leaky context if available
    for extra_col in [
        "route_path",
        "corridor_avg_impact",
        "junction_avg_impact"
    ]:
        if extra_col in df.columns and extra_col not in DURATION_FEATURES:
            DURATION_FEATURES.append(extra_col)

    DURATION_FEATURES = [c for c in DURATION_FEATURES if c in duration_df.columns]

    print("\nDuration features used:", len(DURATION_FEATURES))
    print(DURATION_FEATURES)

    X_duration = duration_df[DURATION_FEATURES].copy()
    y_duration_raw = duration_df["duration_minutes"].astype(float)
    y_duration = np.log1p(y_duration_raw)

    X_duration = fill_object_cols(X_duration)
    X_duration = fill_numeric_cols(X_duration)

    X_train_d, X_test_d, y_train_d, y_test_d, y_train_raw, y_test_raw = train_test_split(
        X_duration,
        y_duration,
        y_duration_raw,
        test_size=0.20,
        random_state=RANDOM_STATE
    )

    duration_feature_cols = X_train_d.columns.tolist()
    duration_cat_cols = [c for c in duration_feature_cols if X_train_d[c].dtype == "object"]
    duration_cat_idx = [duration_feature_cols.index(c) for c in duration_cat_cols]

    duration_model = CatBoostRegressor(
        iterations=700,
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=RANDOM_STATE,
        verbose=100,
        l2_leaf_reg=8
    )

    duration_model.fit(
        X_train_d,
        y_train_d,
        cat_features=duration_cat_idx,
        eval_set=(X_test_d, y_test_d),
        use_best_model=True
    )

    pred_log = duration_model.predict(X_test_d)
    pred_minutes = np.expm1(pred_log)
    pred_minutes = np.clip(pred_minutes, 0, MAX_DURATION)

    mae = mean_absolute_error(y_test_raw, pred_minutes)
    rmse = np.sqrt(mean_squared_error(y_test_raw, pred_minutes))
    r2 = r2_score(y_test_raw, pred_minutes)

    print("\nDuration Regression Metrics:")
    print("MAE :", mae)
    print("RMSE:", rmse)
    print("R2  :", r2)

    duration_importance = pd.DataFrame({
        "feature": duration_feature_cols,
        "importance": duration_model.get_feature_importance()
    }).sort_values("importance", ascending=False)

    print("\nTop 25 Duration Features:")
    print(duration_importance.head(25))

    joblib.dump(duration_model, os.path.join(MODEL_DIR, "duration_regressor_catboost.pkl"))
    joblib.dump(duration_feature_cols, os.path.join(MODEL_DIR, "duration_feature_columns.pkl"))
    joblib.dump(duration_cat_cols, os.path.join(MODEL_DIR, "duration_categorical_feature_names.pkl"))
    duration_importance.to_csv(os.path.join(MODEL_DIR, "duration_feature_importance.csv"), index=False)

# ============================================================
# 4) SAVE META
# ============================================================
model_meta = {
    "input_file": INPUT_FILE,
    "random_state": RANDOM_STATE,
    "safe_pre_event_features": SAFE_PRE_EVENT_FEATURES,
    "duration_min": MIN_DURATION,
    "duration_max": MAX_DURATION,
    "num_rows_total": int(len(df)),
    "closure_target_distribution": df["requires_road_closure_flag"].value_counts().to_dict()
}
joblib.dump(model_meta, os.path.join(MODEL_DIR, "model_meta.pkl"))

# ============================================================
# DONE
# ============================================================
print("\n" + "=" * 100)
print("FINAL SAFE STEP-2 TRAINING COMPLETED")
print("=" * 100)
print("Saved in:", MODEL_DIR)
print("""
Files expected:
- closure_classifier_catboost.pkl
- closure_feature_columns.pkl
- closure_categorical_feature_names.pkl
- closure_feature_importance.csv

- duration_regressor_catboost.pkl
- duration_feature_columns.pkl
- duration_categorical_feature_names.pkl
- duration_feature_importance.csv

- model_meta.pkl
""")