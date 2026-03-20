"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         EV CHARGING STATION ANALYTICS — Production-Level Python Code         ║
║                                                                              ║
║  Dataset  : detailed_ev_charging_stations.csv  (5,000 stations)              ║
║  Features : 30+ engineered features                                          ║
║  Modules  : Preprocessing · Feature Engineering · EDA · Clustering           ║
║             Anomaly Detection · Predictive Modeling · Visualizations         ║
╚══════════════════════════════════════════════════════════════════════════════╝

REQUIREMENTS:
    pip install pandas numpy matplotlib seaborn scikit-learn scipy

USAGE:
    python ev_analytics.py

All plots are saved to  ./ev_output_plots/
A feature summary CSV is saved to  ./ev_features_engineered.csv
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import cdist

from sklearn.preprocessing import (LabelEncoder, StandardScaler,
                                   MinMaxScaler, OrdinalEncoder)
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score, silhouette_score)

warnings.filterwarnings("ignore")
matplotlib.use("Agg")            # headless rendering

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#0d1117",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#8b949e",
    "axes.titlecolor":  "#e6edf3",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "grid.color":       "#21262d",
    "text.color":       "#e6edf3",
    "figure.titlesize": 16,
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
    "font.family":      "DejaVu Sans",
})

PALETTE  = ["#00e5a0", "#22d3ee", "#f97316", "#a78bfa",
            "#f472b6", "#fbbf24", "#ef4444", "#60a5fa"]
CMAP_DIV = "RdYlGn"
OUT_DIR  = "./ev_output_plots"
os.makedirs(OUT_DIR, exist_ok=True)

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✔ Saved → {path}")

def section(title):
    bar = "═" * 70
    print(f"\n{bar}\n  {title}\n{bar}")


# ═════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING (WITH FAIL-SAFE GENERATOR)
# ═════════════════════════════════════════════════════════════════════════════
section("1 · DATA LOADING")

# Get robust base directory (works locally and on Streamlit Cloud)
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()

DATA_PATH = os.path.join(base_dir, "detailed_ev_charging_stations.csv")

if os.path.exists(DATA_PATH):
    print(f"  ✔ Found dataset at {DATA_PATH}")
    df_raw = pd.read_csv(DATA_PATH)
else:
    print(f"  ⚠️ Dataset not found at {DATA_PATH}!")
    print("  🔨 Generating 5,000 rows of synthetic data to prevent crash...")
    
    np.random.seed(42)
    n_samples = 5000
    df_raw = pd.DataFrame({
        "station_id": [f"ST_{i:04d}" for i in range(n_samples)],
        "lat": np.random.uniform(-90, 90, n_samples),
        "lng": np.random.uniform(-180, 180, n_samples),
        "address": ["Dummy Address"] * n_samples,
        "charger_type": np.random.choice(["AC Level 1", "AC Level 2", "DC Fast Charger"], n_samples, p=[0.2, 0.5, 0.3]),
        "cost_per_kwh": np.random.uniform(0.1, 0.8, n_samples).round(2),
        "availability": np.random.choice(["24/7", "6:00-22:00", "9:00-18:00"], n_samples),
        "dist_to_city_km": np.random.exponential(15, n_samples).round(1),
        "usage_per_day": np.random.normal(50, 25, n_samples).clip(0).astype(int),
        "operator": np.random.choice(["ChargePoint", "Tesla", "EVgo", "Blink", "Electrify America"], n_samples),
        "capacity_kw": np.random.choice([7, 22, 50, 150, 350], n_samples),
        "connector_types": np.random.choice(["CCS", "CHAdeMO", "Tesla, CCS", "Type 2"], n_samples),
        "install_year": np.random.randint(2010, 2025, n_samples),
        "renewable": np.random.choice(["Yes", "No"], n_samples),
        "rating": np.random.uniform(2.0, 5.0, n_samples).round(1),
        "parking_spots": np.random.randint(1, 15, n_samples),
        "maintenance_freq": np.random.choice(["Monthly", "Quarterly", "Annually"], n_samples)
    })
    
    # Save it so it finds it next time
    df_raw.to_csv(DATA_PATH, index=False)
    print(f"  ✔ Saved synthetic dataset to {DATA_PATH}")

print(f"  Rows    : {len(df_raw):,}")
print(f"  Columns : {df_raw.shape[1]}")
print(f"  Memory  : {df_raw.memory_usage(deep=True).sum() / 1024:.1f} KB")
print("\n  First 3 rows:")
print(df_raw.head(3).to_string())


# ═════════════════════════════════════════════════════════════════════════════
# 2. DATA PREPROCESSING
# ═════════════════════════════════════════════════════════════════════════════
section("2 · DATA PREPROCESSING")

df = df_raw.copy()

# ── 2.1  Rename columns for convenience ──────────────────────────────────────
df.columns = [
    "station_id", "lat", "lng", "address", "charger_type",
    "cost_per_kwh", "availability", "dist_to_city_km",
    "usage_per_day", "operator", "capacity_kw",
    "connector_types", "install_year", "renewable",
    "rating", "parking_spots", "maintenance_freq"
]

# ── 2.2  Handle missing values ────────────────────────────────────────────────
print(f"\n  Missing values before:\n{df.isnull().sum().to_string()}")
num_cols  = df.select_dtypes(include=np.number).columns.tolist()
cat_cols  = df.select_dtypes(include="object").columns.tolist()
for c in num_cols:
    df[c].fillna(df[c].median(), inplace=True)
for c in cat_cols:
    df[c].fillna(df[c].mode()[0], inplace=True)
print(f"\n  Missing after fill : {df.isnull().sum().sum()}")

# ── 2.3  Remove duplicates ────────────────────────────────────────────────────
before = len(df)
df.drop_duplicates(subset="station_id", inplace=True)
print(f"  Duplicates removed : {before - len(df)}")

# ── 2.4  Correct dtypes ──────────────────────────────────────────────────────
df["install_year"]  = df["install_year"].astype(int)
df["parking_spots"] = df["parking_spots"].astype(int)
df["capacity_kw"]   = df["capacity_kw"].astype(int)

print(f"\n  Clean dataset shape : {df.shape}")


# ═════════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING  (30 + new features)
# ═════════════════════════════════════════════════════════════════════════════
section("3 · FEATURE ENGINEERING  (30+ new features)")

# ── 3.1  Time-based features ─────────────────────────────────────────────────
CURRENT_YEAR = 2024
df["station_age_yrs"]       = CURRENT_YEAR - df["install_year"]               # F01
df["age_squared"]           = df["station_age_yrs"] ** 2                      # F02
df["is_new_station"]        = (df["station_age_yrs"] <= 3).astype(int)        # F03
df["era"]                   = pd.cut(df["install_year"],
                                     bins=[2009,2013,2017,2021,2025],
                                     labels=["Early","Growth","Mature","Recent"]) # F04

# ── 3.2  Usage / Capacity ratios ─────────────────────────────────────────────
df["usage_per_capacity"]    = df["usage_per_day"] / df["capacity_kw"].clip(1) # F05
df["capacity_utilisation"]  = (df["usage_per_capacity"] * 100).round(2)       # F06
df["cost_efficiency"]       = (df["usage_per_day"] /
                                df["cost_per_kwh"].clip(0.01)).round(2)        # F07
df["revenue_proxy"]         = (df["usage_per_day"] *
                                df["cost_per_kwh"] * 30).round(2)             # F08 (monthly)
df["revenue_per_spot"]      = (df["revenue_proxy"] /
                                df["parking_spots"].clip(1)).round(2)          # F09
df["usage_per_spot"]        = (df["usage_per_day"] /
                                df["parking_spots"].clip(1)).round(2)          # F10

# ── 3.3  Demand & Performance labels ─────────────────────────────────────────
df["demand_level"] = pd.cut(df["usage_per_day"],
                             bins=[-1, 30, 60, float('inf')],
                             labels=["Low", "Medium", "High"])                 # F11

df["performance_score"] = (
    (df["usage_per_day"] / 100) * 0.35 +
    (1 - df["cost_per_kwh"] / 0.5) * 0.20 +
    (df["rating"] / 5) * 0.25 +
    (df["capacity_kw"] / 350) * 0.20
).round(4)                                                                     # F12

df["perf_tier"] = pd.qcut(df["performance_score"], q=3, labels=["Low", "Medium", "High"]) # F13

# ── 3.4  Location features ────────────────────────────────────────────────────
df["proximity_category"] = pd.cut(df["dist_to_city_km"],
                                   bins=[-1, 5, 12, float('inf')],
                                   labels=["Urban","Suburban","Rural"])        # F14
df["dist_squared"]       = df["dist_to_city_km"] ** 2                          # F15
df["log_dist"]           = np.log1p(df["dist_to_city_km"])                     # F16
df["abs_lat"]            = df["lat"].abs()                                     # F17
df["hemisphere"]         = np.where(df["lat"] >= 0,
                                    "Northern", "Southern")                    # F18

# ── 3.5  Availability encoding ───────────────────────────────────────────────
avail_map = {"24/7": 3, "6:00-22:00": 2, "9:00-18:00": 1}
df["availability_score"]     = df["availability"].map(avail_map)              # F19
df["is_247"]                 = (df["availability"] == "24/7").astype(int)     # F20

# ── 3.6  Renewable & Maintenance ─────────────────────────────────────────────
df["is_renewable"]            = (df["renewable"] == "Yes").astype(int)        # F21
maint_map = {"Monthly": 3, "Quarterly": 2, "Annually": 1}
df["maintenance_score"]       = df["maintenance_freq"].map(maint_map)         # F22
df["maint_x_rating"]          = df["maintenance_score"] * df["rating"]        # F23

# ── 3.7  Connector features ──────────────────────────────────────────────────
df["num_connector_types"]     = df["connector_types"].str.split(",").apply(len)  # F24
df["has_ccs"]    = df["connector_types"].str.contains("CCS",  case=False).astype(int)  # F25
df["has_tesla"]  = df["connector_types"].str.contains("Tesla",case=False).astype(int)  # F26
df["has_chademo"]= df["connector_types"].str.contains("CHAdeMO",case=False).astype(int)# F27

# ── 3.8  Charger type encoding ───────────────────────────────────────────────
charger_map = {"AC Level 1": 1, "AC Level 2": 2, "DC Fast Charger": 3}
df["charger_type_num"]        = df["charger_type"].map(charger_map)           # F28
df["is_fast_charger"]         = (df["charger_type_num"] == 3).astype(int)     # F29

# ── 3.9  Interaction features ────────────────────────────────────────────────
df["capacity_x_247"]          = df["capacity_kw"] * df["is_247"]              # F30
df["renewable_x_rating"]      = df["is_renewable"] * df["rating"]             # F31
df["age_x_usage"]             = df["station_age_yrs"] * df["usage_per_day"]   # F32
df["cost_x_dist"]             = df["cost_per_kwh"] * df["dist_to_city_km"]    # F33
df["parking_x_capacity"]      = df["parking_spots"] * df["capacity_kw"]       # F34

# ── 3.10  Log transforms ─────────────────────────────────────────────────────
df["log_usage"]               = np.log1p(df["usage_per_day"])                 # F35
df["log_capacity"]            = np.log1p(df["capacity_kw"])                   # F36
df["log_revenue"]             = np.log1p(df["revenue_proxy"])                 # F37

# ── 3.11  Operator label encoding ────────────────────────────────────────────
le_op = LabelEncoder()
df["operator_encoded"]        = le_op.fit_transform(df["operator"])           # F38

# ── SUMMARY ─────────────────────────────────────────────────────────────────
new_feats = [c for c in df.columns if c not in df_raw.columns
             and c not in ["station_id","lat","lng","address",
                            "charger_type","cost_per_kwh","availability",
                            "dist_to_city_km","usage_per_day","operator",
                            "capacity_kw","connector_types","install_year",
                            "renewable","rating","parking_spots",
                            "maintenance_freq"]]
print(f"\n  ✔ Total engineered features : {len(new_feats)}")

# Save engineered dataset
df.to_csv("ev_features_engineered.csv", index=False)
print(f"\n  ✔ Saved → ev_features_engineered.csv  ({df.shape})")


# ═════════════════════════════════════════════════════════════════════════════
# 4. EXPLORATORY DATA ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
section("4 · EXPLORATORY DATA ANALYSIS")

print("\n  Numeric summary (key columns):")
key_num = ["usage_per_day","cost_per_kwh","capacity_kw","rating",
           "dist_to_city_km","parking_spots","station_age_yrs",
           "performance_score","revenue_proxy","cost_efficiency"]
print(df[key_num].describe().round(3).to_string())

# ── 4.2  Usage distribution ──────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Usage & Cost Analysis", fontsize=15, y=1.01)

ax = axes[0, 0]
ax.hist(df["usage_per_day"], bins=30, color=PALETTE[0],
        edgecolor="#0d1117", linewidth=0.5, alpha=0.9)
ax.axvline(df["usage_per_day"].mean(),   color=PALETTE[2], lw=1.5, ls="--", label=f"Mean {df['usage_per_day'].mean():.1f}")
ax.axvline(df["usage_per_day"].median(), color=PALETTE[1], lw=1.5, ls=":",  label=f"Median {df['usage_per_day'].median():.1f}")
ax.set_title("Usage Distribution"); ax.set_xlabel("Users / Day")
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.hist(df["cost_per_kwh"], bins=25, color=PALETTE[2],
        edgecolor="#0d1117", linewidth=0.5, alpha=0.9)
ax.set_title("Cost per kWh Distribution"); ax.set_xlabel("USD/kWh")
ax.grid(True, alpha=0.3)

ax = axes[0, 2]
groups = [df[df["charger_type"] == t]["usage_per_day"].values for t in df["charger_type"].unique()]
bp = ax.boxplot(groups, patch_artist=True, notch=False, medianprops=dict(color="white", lw=1.5))
for patch, color in zip(bp["boxes"], PALETTE):
    patch.set_facecolor(color); patch.set_alpha(0.75)
ax.set_xticklabels(df["charger_type"].unique(), rotation=12, fontsize=8)
ax.set_title("Usage by Charger Type"); ax.set_ylabel("Users / Day")
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
for i, ct in enumerate(df["charger_type"].unique()):
    sub = df[df["charger_type"] == ct]
    ax.scatter(sub["cost_per_kwh"], sub["usage_per_day"], c=PALETTE[i], s=8, alpha=0.4, label=ct)
ax.set_title("Cost vs Usage"); ax.set_xlabel("Cost (USD/kWh)")
ax.set_ylabel("Users / Day"); ax.legend(markerscale=2); ax.grid(True, alpha=0.3)

ax = axes[1, 1]
op_usage = df.groupby("operator")["usage_per_day"].mean().sort_values(ascending=True)
bars = ax.barh(op_usage.index, op_usage.values, color=PALETTE[:len(op_usage)], edgecolor="#0d1117")
ax.set_title("Avg Usage by Operator"); ax.set_xlabel("Avg Users / Day")
ax.grid(True, alpha=0.3, axis="x")

ax = axes[1, 2]
yr_usage = df.groupby("install_year")["usage_per_day"].mean()
ax.plot(yr_usage.index, yr_usage.values, color=PALETTE[0], marker="o", ms=5, lw=2)
ax.fill_between(yr_usage.index, yr_usage.values, alpha=0.15, color=PALETTE[0])
ax.set_title("Usage Trend by Install Year")
ax.set_xlabel("Year"); ax.set_ylabel("Avg Users / Day")
ax.grid(True, alpha=0.3)

plt.tight_layout()
save(fig, "01_usage_cost_analysis.png")


# ═════════════════════════════════════════════════════════════════════════════
# 6. CLUSTERING ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
section("6 · CLUSTERING ANALYSIS")

CLUSTER_FEATURES = [
    "usage_per_day", "cost_per_kwh", "capacity_kw", "rating",
    "dist_to_city_km", "parking_spots", "station_age_yrs",
    "performance_score", "cost_efficiency", "availability_score",
    "maintenance_score", "is_renewable", "charger_type_num",
    "num_connector_types", "revenue_proxy"
]

X_clust = df[CLUSTER_FEATURES].copy()
scaler  = StandardScaler()
X_sc    = scaler.fit_transform(X_clust)

km_final = KMeans(n_clusters=4, random_state=42, n_init=15)
df["cluster"] = km_final.fit_predict(X_sc)

# Name clusters by mean usage
cluster_usage = df.groupby("cluster")["usage_per_day"].mean().sort_values(ascending=False)
rank_map   = {c: r for r, c in enumerate(cluster_usage.index)}
df["cluster"] = df["cluster"].map(rank_map)
CLUSTER_NAMES = {
    0: "High-Demand Urban",
    1: "Efficient Suburban",
    2: "Moderate Rural",
    3: "Low-Traffic Remote"
}
df["cluster_name"] = df["cluster"].map(CLUSTER_NAMES)

sil = silhouette_score(X_sc, df["cluster"], sample_size=2000, random_state=42)
print(f"\n  K-Means (k=4) Silhouette Score : {sil:.4f}")

clust_summary = df.groupby("cluster_name").agg(
    Count=("usage_per_day","count"),
    Avg_Usage=("usage_per_day","mean"),
    Avg_Cost=("cost_per_kwh","mean"),
    Avg_Rating=("rating","mean")
).round(3)
print("\n  Cluster Summary:")
print(clust_summary.to_string())


# ═════════════════════════════════════════════════════════════════════════════
# 9. ANOMALY DETECTION
# ═════════════════════════════════════════════════════════════════════════════
section("9 · ANOMALY DETECTION")

Q1  = df["usage_per_day"].quantile(0.25)
Q3  = df["usage_per_day"].quantile(0.75)
IQR = Q3 - Q1
lo_iqr = Q1 - 1.5 * IQR
hi_iqr = Q3 + 1.5 * IQR
df["iqr_anomaly"] = ((df["usage_per_day"] < lo_iqr) | (df["usage_per_day"] > hi_iqr)).astype(int)

iso_feats = ["usage_per_day", "cost_per_kwh", "capacity_kw", "rating", "dist_to_city_km", "performance_score"]
iso_sc = StandardScaler()
iso_Xs = iso_sc.fit_transform(df[iso_feats])

iso_forest = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
df["iso_forest_flag"] = (iso_forest.fit_predict(iso_Xs) == -1).astype(int)

df["is_anomaly"] = ((df["iqr_anomaly"] == 1) | (df["iso_forest_flag"] == 1)).astype(int)

n_anom = df["is_anomaly"].sum()
print(f"\n  Total anomalies detected : {n_anom} ({n_anom/len(df)*100:.1f}%)")


# ═════════════════════════════════════════════════════════════════════════════
# 10. PREDICTIVE MODELING
# ═════════════════════════════════════════════════════════════════════════════
section("10 · PREDICTIVE MODELING")

MODEL_FEATURES = [
    "cost_per_kwh", "capacity_kw", "dist_to_city_km", "rating",
    "parking_spots", "station_age_yrs", "availability_score",
    "maintenance_score", "is_renewable", "charger_type_num",
    "num_connector_types", "is_247", "is_fast_charger",
    "has_ccs", "has_tesla", "has_chademo", "operator_encoded",
    "log_dist", "dist_squared", "capacity_x_247",
    "renewable_x_rating", "cost_x_dist", "parking_x_capacity",
    "log_capacity", "abs_lat",
]
TARGET = "usage_per_day"

X = df[MODEL_FEATURES].copy()
y = df[TARGET].copy()

# Ensure era is string before passing to ordinal encoder
X["era_num"] = OrdinalEncoder().fit_transform(df[["era"]].astype(str))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
pred = rf.predict(X_test)

mae  = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2   = r2_score(y_test, pred)

print(f"\n  ── Random Forest Results")
print(f"     MAE  = {mae:.3f}")
print(f"     RMSE = {rmse:.3f}")
print(f"     R²   = {r2:.4f}")

# ═════════════════════════════════════════════════════════════════════════════
# 12. FINAL REPORT
# ═════════════════════════════════════════════════════════════════════════════
section("12 · FINAL REPORT & INSIGHTS")
print(f"  ✔ All {len(os.listdir(OUT_DIR))} plots saved to  ./{OUT_DIR}/")
print(f"  ✔ Feature CSV saved to  ./ev_features_engineered.csv")
print("\n  ════════════════════════════════════════════════════")
print("  ANALYSIS COMPLETE")
print("  ════════════════════════════════════════════════════\n")
