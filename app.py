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
from sklearn.pipeline import Pipeline

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
# 1. DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════
section("1 · DATA LOADING")

DATA_PATH = "detailed_ev_charging_stations.csv"
# fallback – look in uploads folder if not in cwd
if not os.path.exists(DATA_PATH):
    DATA_PATH = "/mnt/user-data/uploads/detailed_ev_charging_stations.csv"

df_raw = pd.read_csv(DATA_PATH)
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
                                     bins=[2009,2013,2017,2021,2024],
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
                             bins=[-1, 30, 60, 100],
                             labels=["Low", "Medium", "High"])                 # F11

df["performance_score"] = (
    (df["usage_per_day"] / 100) * 0.35 +
    (1 - df["cost_per_kwh"] / 0.5) * 0.20 +
    (df["rating"] / 5) * 0.25 +
    (df["capacity_kw"] / 350) * 0.20
).round(4)                                                                     # F12

df["perf_tier"] = pd.cut(df["performance_score"],
                          bins=3,
                          labels=["Low", "Medium", "High"])                    # F13

# ── 3.4  Location features ────────────────────────────────────────────────────
df["proximity_category"] = pd.cut(df["dist_to_city_km"],
                                   bins=[-1, 5, 12, 20],
                                   labels=["Urban","Suburban","Rural"])        # F14
df["dist_squared"]       = df["dist_to_city_km"] ** 2                         # F15
df["log_dist"]           = np.log1p(df["dist_to_city_km"])                    # F16
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
for i, f in enumerate(new_feats, 1):
    print(f"    F{i:02d}. {f}")

# Save engineered dataset
df.to_csv("ev_features_engineered.csv", index=False)
print(f"\n  ✔ Saved → ev_features_engineered.csv  ({df.shape})")


# ═════════════════════════════════════════════════════════════════════════════
# 4. EXPLORATORY DATA ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
section("4 · EXPLORATORY DATA ANALYSIS")

# ── 4.1  Descriptive statistics ──────────────────────────────────────────────
print("\n  Numeric summary (key columns):")
key_num = ["usage_per_day","cost_per_kwh","capacity_kw","rating",
           "dist_to_city_km","parking_spots","station_age_yrs",
           "performance_score","revenue_proxy","cost_efficiency"]
print(df[key_num].describe().round(3).to_string())

# ── 4.2  Usage distribution ──────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Usage & Cost Analysis", fontsize=15, y=1.01)

# Histogram – usage
ax = axes[0, 0]
ax.hist(df["usage_per_day"], bins=30, color=PALETTE[0],
        edgecolor="#0d1117", linewidth=0.5, alpha=0.9)
ax.axvline(df["usage_per_day"].mean(),   color=PALETTE[2], lw=1.5, ls="--", label=f"Mean {df['usage_per_day'].mean():.1f}")
ax.axvline(df["usage_per_day"].median(), color=PALETTE[1], lw=1.5, ls=":",  label=f"Median {df['usage_per_day'].median():.1f}")
ax.set_title("Usage Distribution"); ax.set_xlabel("Users / Day")
ax.legend(); ax.grid(True, alpha=0.3)

# Histogram – cost
ax = axes[0, 1]
ax.hist(df["cost_per_kwh"], bins=25, color=PALETTE[2],
        edgecolor="#0d1117", linewidth=0.5, alpha=0.9)
ax.set_title("Cost per kWh Distribution"); ax.set_xlabel("USD/kWh")
ax.grid(True, alpha=0.3)

# Boxplot – usage by charger type
ax = axes[0, 2]
groups = [df[df["charger_type"] == t]["usage_per_day"].values
          for t in df["charger_type"].unique()]
bp = ax.boxplot(groups, patch_artist=True, notch=False,
                medianprops=dict(color="white", lw=1.5))
for patch, color in zip(bp["boxes"], PALETTE):
    patch.set_facecolor(color); patch.set_alpha(0.75)
ax.set_xticklabels(df["charger_type"].unique(), rotation=12, fontsize=8)
ax.set_title("Usage by Charger Type"); ax.set_ylabel("Users / Day")
ax.grid(True, alpha=0.3)

# Scatter – cost vs usage
ax = axes[1, 0]
for i, ct in enumerate(df["charger_type"].unique()):
    sub = df[df["charger_type"] == ct]
    ax.scatter(sub["cost_per_kwh"], sub["usage_per_day"],
               c=PALETTE[i], s=8, alpha=0.4, label=ct)
ax.set_title("Cost vs Usage"); ax.set_xlabel("Cost (USD/kWh)")
ax.set_ylabel("Users / Day"); ax.legend(markerscale=2); ax.grid(True, alpha=0.3)

# Bar – avg usage by operator
ax = axes[1, 1]
op_usage = df.groupby("operator")["usage_per_day"].mean().sort_values(ascending=True)
bars = ax.barh(op_usage.index, op_usage.values,
               color=PALETTE[:len(op_usage)], edgecolor="#0d1117")
ax.set_title("Avg Usage by Operator"); ax.set_xlabel("Avg Users / Day")
ax.grid(True, alpha=0.3, axis="x")
for bar, val in zip(bars, op_usage.values):
    ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}", va="center", fontsize=8, color="#e6edf3")

# Trend – avg usage by install year
ax = axes[1, 2]
yr_usage = df.groupby("install_year")["usage_per_day"].mean()
ax.plot(yr_usage.index, yr_usage.values,
        color=PALETTE[0], marker="o", ms=5, lw=2)
ax.fill_between(yr_usage.index, yr_usage.values,
                alpha=0.15, color=PALETTE[0])
ax.set_title("Usage Trend by Install Year")
ax.set_xlabel("Year"); ax.set_ylabel("Avg Users / Day")
ax.grid(True, alpha=0.3)

plt.tight_layout()
save(fig, "01_usage_cost_analysis.png")

# ── 4.3  Feature distribution grid ──────────────────────────────────────────
fig, axes = plt.subplots(3, 4, figsize=(18, 12))
fig.suptitle("Engineered Feature Distributions", fontsize=15)
feats_to_plot = ["performance_score","cost_efficiency","revenue_proxy",
                 "capacity_utilisation","usage_per_spot","revenue_per_spot",
                 "age_x_usage","log_usage","log_capacity","log_revenue",
                 "cost_x_dist","maint_x_rating"]
for ax, feat in zip(axes.flat, feats_to_plot):
    vals = df[feat].dropna()
    ax.hist(vals, bins=28, color=PALETTE[feats_to_plot.index(feat) % 8],
            edgecolor="#0d1117", linewidth=0.4, alpha=0.85)
    ax.set_title(feat.replace("_"," ").title(), fontsize=9)
    ax.grid(True, alpha=0.25)
plt.tight_layout()
save(fig, "02_engineered_feature_distributions.png")

# ── 4.4  Categorical analysis ────────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
fig.suptitle("Categorical Feature Analysis")

cats_info = [
    ("charger_type",   "Charger Type Counts",       "usage_per_day",  "Avg Usage by Charger Type"),
    ("operator",       "Station Count by Operator", "rating",         "Avg Rating by Operator"),
    ("demand_level",   "Demand Level Distribution", "cost_per_kwh",   "Avg Cost by Demand Level"),
]
for row, (cat1, t1, cat2, t2) in enumerate(cats_info):
    ax = axes[row, 0]
    counts = df[cat1].value_counts()
    ax.bar(counts.index, counts.values,
           color=PALETTE[:len(counts)], edgecolor="#0d1117")
    ax.set_title(t1); ax.set_xticklabels(counts.index, rotation=20, ha="right")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[row, 1]
    means = df.groupby(cat1, observed=True)[cat2].mean().sort_values(ascending=False)
    ax.bar(means.index, means.values,
           color=PALETTE[3:3+len(means)], edgecolor="#0d1117")
    ax.set_title(t2); ax.set_xticklabels(means.index, rotation=20, ha="right")
    ax.grid(True, alpha=0.3, axis="y")

# Pie – renewable
ax = axes[0, 2]
renew = df["renewable"].value_counts()
ax.pie(renew.values, labels=renew.index,
       colors=[PALETTE[0], PALETTE[6]],
       autopct="%1.1f%%", startangle=90,
       textprops={"color": "#e6edf3", "fontsize": 10})
ax.set_title("Renewable Energy Source")

# Pie – availability
ax = axes[1, 2]
av = df["availability"].value_counts()
ax.pie(av.values, labels=av.index,
       colors=PALETTE[:3],
       autopct="%1.1f%%", startangle=90,
       textprops={"color": "#e6edf3", "fontsize": 10})
ax.set_title("Availability Hours")

# Pie – maintenance
ax = axes[2, 2]
mf = df["maintenance_freq"].value_counts()
ax.pie(mf.values, labels=mf.index,
       colors=PALETTE[3:6],
       autopct="%1.1f%%", startangle=90,
       textprops={"color": "#e6edf3", "fontsize": 10})
ax.set_title("Maintenance Frequency")

plt.tight_layout()
save(fig, "03_categorical_analysis.png")


# ── 4.5  Correlation heatmap (all numeric features) ──────────────────────────
section("4.5 · Correlation Heatmap")

num_features = [
    "usage_per_day","cost_per_kwh","capacity_kw","rating",
    "dist_to_city_km","parking_spots","station_age_yrs",
    "usage_per_capacity","capacity_utilisation","cost_efficiency",
    "revenue_proxy","revenue_per_spot","usage_per_spot",
    "performance_score","availability_score","maintenance_score",
    "num_connector_types","charger_type_num","is_247","is_renewable",
    "is_fast_charger","maint_x_rating","age_x_usage","cost_x_dist",
    "parking_x_capacity","log_usage","log_capacity","log_revenue",
    "dist_squared","log_dist"
]

corr_matrix = df[num_features].corr()
fig, ax = plt.subplots(figsize=(20, 16))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
            cmap="RdYlGn", vmin=-1, vmax=1,
            linewidths=0.3, linecolor="#1e2d4a",
            annot_kws={"size": 6},
            cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title("Full Feature Correlation Matrix (30 features)", fontsize=14, pad=20)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
plt.tight_layout()
save(fig, "04_correlation_heatmap.png")

# Top correlations with target
target_corr = corr_matrix["usage_per_day"].drop("usage_per_day").sort_values(key=abs, ascending=False)
print("\n  Top 15 features correlated with usage_per_day:")
print(target_corr.head(15).round(4).to_string())


# ─────────────────────────────────────────────────────────────────────────────
# 5. ADVANCED VISUAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
section("5 · ADVANCED VISUAL ANALYSIS")

# ── 5.1  Multi-variable pairplot (sample) ────────────────────────────────────
sample = df.sample(600, random_state=42)
pair_vars = ["usage_per_day","cost_per_kwh","capacity_kw",
             "rating","dist_to_city_km","performance_score"]
fig = plt.figure(figsize=(16, 14))
fig.suptitle("Pairplot — Key Variables Coloured by Charger Type", fontsize=13)
charger_types = df["charger_type"].unique()
n = len(pair_vars)
for i, v1 in enumerate(pair_vars):
    for j, v2 in enumerate(pair_vars):
        ax = fig.add_subplot(n, n, i * n + j + 1)
        if i == j:
            for k, ct in enumerate(charger_types):
                vals = sample[sample["charger_type"] == ct][v1]
                ax.hist(vals, bins=20, alpha=0.55,
                        color=PALETTE[k], edgecolor="none")
        else:
            for k, ct in enumerate(charger_types):
                sub = sample[sample["charger_type"] == ct]
                ax.scatter(sub[v2], sub[v1],
                           s=4, alpha=0.3, color=PALETTE[k])
        ax.set_xlabel(v2.replace("_", "\n") if j == n-1 else "", fontsize=6)
        ax.set_ylabel(v1.replace("_", "\n") if j == 0 else "", fontsize=6)
        ax.tick_params(labelsize=5)
patches = [mpatches.Patch(color=PALETTE[k], label=ct)
           for k, ct in enumerate(charger_types)]
fig.legend(handles=patches, loc="lower right",
           fontsize=9, title="Charger Type")
plt.tight_layout()
save(fig, "05_pairplot_key_variables.png")

# ── 5.2  Performance score deep-dive ─────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Performance Score Deep Dive")

ax = axes[0, 0]
for k, ct in enumerate(df["charger_type"].unique()):
    sub = df[df["charger_type"] == ct]["performance_score"]
    ax.hist(sub, bins=25, alpha=0.6, color=PALETTE[k], label=ct)
ax.set_title("Performance by Charger Type")
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
op_perf = df.groupby("operator")["performance_score"].mean().sort_values()
ax.barh(op_perf.index, op_perf.values,
        color=PALETTE[:len(op_perf)], edgecolor="#0d1117")
ax.set_title("Avg Performance by Operator"); ax.grid(True, alpha=0.3, axis="x")

ax = axes[0, 2]
prox_perf = df.groupby("proximity_category",observed=True)["performance_score"].mean()
ax.bar(prox_perf.index, prox_perf.values,
       color=[PALETTE[0],PALETTE[2],PALETTE[3]], edgecolor="#0d1117")
ax.set_title("Avg Performance by Proximity"); ax.grid(True, alpha=0.3, axis="y")

ax = axes[1, 0]
yr_perf = df.groupby("install_year")["performance_score"].mean()
ax.plot(yr_perf.index, yr_perf.values, color=PALETTE[4],
        marker="s", ms=4, lw=2)
ax.fill_between(yr_perf.index, yr_perf.values, alpha=0.12, color=PALETTE[4])
ax.set_title("Avg Performance by Year"); ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.scatter(df["cost_per_kwh"], df["performance_score"],
           c=df["usage_per_day"], cmap="RdYlGn",
           s=6, alpha=0.4)
ax.set_title("Cost vs Performance (colour=usage)")
ax.set_xlabel("Cost USD/kWh"); ax.set_ylabel("Performance Score")
ax.grid(True, alpha=0.3)

ax = axes[1, 2]
renew_perf = df.groupby("renewable")["performance_score"].mean()
ax.bar(renew_perf.index, renew_perf.values,
       color=[PALETTE[0], PALETTE[6]], edgecolor="#0d1117", width=0.4)
ax.set_title("Avg Performance: Renewable vs Not"); ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
save(fig, "06_performance_deep_dive.png")

# ── 5.3  Revenue analysis ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Revenue & Cost Efficiency Analysis")

ax = axes[0]
ax.hist(df["revenue_proxy"], bins=30, color=PALETTE[5],
        edgecolor="#0d1117", alpha=0.9)
ax.axvline(df["revenue_proxy"].mean(), color=PALETTE[2], lw=2,
           ls="--", label=f"Mean ${df['revenue_proxy'].mean():.0f}")
ax.set_title("Monthly Revenue Proxy"); ax.set_xlabel("USD/month")
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
op_rev = df.groupby("operator")["revenue_proxy"].mean().sort_values(ascending=False)
ax.bar(op_rev.index, op_rev.values, color=PALETTE[:len(op_rev)], edgecolor="#0d1117")
ax.set_title("Avg Monthly Revenue by Operator")
ax.set_xticklabels(op_rev.index, rotation=20, ha="right")
ax.grid(True, alpha=0.3, axis="y")

ax = axes[2]
ax.scatter(df["cost_efficiency"], df["revenue_proxy"],
           c=df["charger_type_num"], cmap="plasma",
           s=8, alpha=0.4)
ax.set_title("Cost Efficiency vs Revenue")
ax.set_xlabel("Cost Efficiency (usage/cost)"); ax.set_ylabel("Revenue Proxy")
ax.grid(True, alpha=0.3)

plt.tight_layout()
save(fig, "07_revenue_analysis.png")


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

# ── 6.1  Elbow + Silhouette ───────────────────────────────────────────────────
print("\n  Computing Elbow & Silhouette scores …")
inertias, silhouettes = [], []
K_range = range(2, 10)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_sc)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_sc, labels, sample_size=2000, random_state=42))
    print(f"    k={k}  inertia={km.inertia_:,.0f}  silhouette={silhouettes[-1]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Optimal Cluster Selection")

ax = axes[0]
ax.plot(K_range, inertias, color=PALETTE[0], marker="o", lw=2)
ax.set_title("Elbow Curve"); ax.set_xlabel("k (Clusters)")
ax.set_ylabel("Inertia"); ax.grid(True, alpha=0.3)

ax = axes[1]
best_k = K_range[int(np.argmax(silhouettes))]
ax.plot(K_range, silhouettes, color=PALETTE[2], marker="s", lw=2)
ax.axvline(best_k, color=PALETTE[6], ls="--", lw=1.5, label=f"Best k={best_k}")
ax.set_title("Silhouette Score"); ax.set_xlabel("k (Clusters)")
ax.set_ylabel("Score"); ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
save(fig, "08_elbow_silhouette.png")
print(f"\n  ✔ Best k by silhouette = {best_k}")

# ── 6.2  K-Means with best k ─────────────────────────────────────────────────
BEST_K = 4   # fixed for interpretability
km_final = KMeans(n_clusters=BEST_K, random_state=42, n_init=15)
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

# Cluster summary
clust_summary = df.groupby("cluster_name").agg(
    Count=("usage_per_day","count"),
    Avg_Usage=("usage_per_day","mean"),
    Avg_Cost=("cost_per_kwh","mean"),
    Avg_Rating=("rating","mean"),
    Avg_Capacity=("capacity_kw","mean"),
    Avg_Performance=("performance_score","mean"),
    Pct_Renewable=("is_renewable","mean"),
    Avg_Revenue=("revenue_proxy","mean")
).round(3)
print("\n  Cluster Summary:")
print(clust_summary.to_string())

# ── 6.3  PCA for visualisation ───────────────────────────────────────────────
pca  = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_sc)
df["pca1"] = X_2d[:, 0]
df["pca2"] = X_2d[:, 1]
var_exp = pca.explained_variance_ratio_ * 100

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("K-Means Clustering (k=4)")

ax = axes[0]
ccolors = [PALETTE[0], PALETTE[2], PALETTE[1], PALETTE[6]]
for ci, (cname, ccolor) in enumerate(zip(CLUSTER_NAMES.values(), ccolors)):
    mask = df["cluster"] == ci
    ax.scatter(df.loc[mask, "pca1"], df.loc[mask, "pca2"],
               s=6, alpha=0.35, color=ccolor, label=cname)
centers_2d = pca.transform(km_final.cluster_centers_[[rank_map[c] for c in range(BEST_K)]])
for ci, (cx, cy) in enumerate(centers_2d):
    ax.scatter(cx, cy, s=200, marker="*",
               color=ccolors[ci], edgecolors="white", linewidths=0.5, zorder=5)
ax.set_title(f"PCA Cluster Scatter  (var={var_exp[0]:.1f}%+{var_exp[1]:.1f}%)")
ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}%)"); ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}%)")
ax.legend(markerscale=3, fontsize=8); ax.grid(True, alpha=0.2)

ax = axes[1]
bars = ax.bar(CLUSTER_NAMES.values(),
              df.groupby("cluster_name")["usage_per_day"].mean()[list(CLUSTER_NAMES.values())],
              color=ccolors, edgecolor="#0d1117")
ax.set_title("Avg Usage per Cluster")
ax.set_xticklabels(CLUSTER_NAMES.values(), rotation=15, ha="right")
ax.set_ylabel("Avg Users / Day"); ax.grid(True, alpha=0.3, axis="y")
for bar, val in zip(bars, df.groupby("cluster_name")["usage_per_day"].mean()[list(CLUSTER_NAMES.values())]):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.5,
            f"{val:.1f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
save(fig, "09_kmeans_clusters.png")

# ── 6.4  Cluster feature heatmap ─────────────────────────────────────────────
feat_means = df.groupby("cluster_name")[CLUSTER_FEATURES].mean()
feat_norm  = (feat_means - feat_means.min()) / (feat_means.max() - feat_means.min() + 1e-9)

fig, ax = plt.subplots(figsize=(16, 5))
sns.heatmap(feat_norm.T, annot=True, fmt=".2f",
            cmap="RdYlGn", linewidths=0.4,
            linecolor="#1e2d4a", ax=ax,
            annot_kws={"size": 8})
ax.set_title("Cluster Feature Profiles (Normalised)", fontsize=13, pad=12)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
plt.tight_layout()
save(fig, "10_cluster_feature_heatmap.png")


# ── 6.5  DBSCAN (Density-based — bonus) ──────────────────────────────────────
section("6.5 · DBSCAN Clustering")

# Use only 2 PCA components for speed + visualisation
db = DBSCAN(eps=0.6, min_samples=8, n_jobs=-1)
db_labels = db.fit_predict(X_2d)
n_db_clust = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise    = (db_labels == -1).sum()
print(f"  DBSCAN clusters found : {n_db_clust}")
print(f"  Noise points          : {n_noise}")

fig, ax = plt.subplots(figsize=(12, 7))
unique_labels = sorted(set(db_labels))
db_colors = plt.cm.Set2(np.linspace(0, 1, max(len(unique_labels), 2)))
for lbl, col in zip(unique_labels, db_colors):
    mask = db_labels == lbl
    label_str = f"Cluster {lbl}" if lbl != -1 else "Noise"
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
               s=4, alpha=0.4, color=col if lbl != -1 else "#555",
               label=f"{label_str} ({mask.sum()})")
ax.set_title(f"DBSCAN Clustering  (eps=0.6, min_samples=8)  →  {n_db_clust} clusters", fontsize=12)
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
ax.legend(markerscale=3, fontsize=7, ncol=3); ax.grid(True, alpha=0.2)
plt.tight_layout()
save(fig, "11_dbscan_clusters.png")


# ═════════════════════════════════════════════════════════════════════════════
# 7. GEOSPATIAL ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
section("7 · GEOSPATIAL ANALYSIS")

fig, axes = plt.subplots(1, 3, figsize=(20, 8))
fig.suptitle("Global EV Station Geospatial Distribution", fontsize=14)

color_maps = [
    ("By Cluster",    "cluster",     ccolors, CLUSTER_NAMES),
    ("By Demand Level", "demand_level",
     {"Low": PALETTE[1], "Medium": PALETTE[2], "High": PALETTE[6]}, None),
    ("By Charger Type", "charger_type",
     {"AC Level 1": PALETTE[1], "AC Level 2": PALETTE[0], "DC Fast Charger": PALETTE[2]}, None),
]

for ax, (title, col, color_info, names) in zip(axes, color_maps):
    if isinstance(color_info, list):
        colors_arr = [color_info[int(c)] for c in df[col]]
    else:
        colors_arr = [color_info.get(str(v), "#555") for v in df[col]]
    ax.scatter(df["lng"], df["lat"],
               c=colors_arr, s=2, alpha=0.35)
    ax.set_title(title); ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_facecolor("#0d1117"); ax.grid(True, alpha=0.15)

plt.tight_layout()
save(fig, "12_geospatial_map.png")

# Density by lat band
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Station Density by Geographic Region")

ax = axes[0]
lat_bins = pd.cut(df["lat"], bins=18)
lat_count = df.groupby(lat_bins, observed=True)["usage_per_day"].agg(["count","mean"])
ax.bar(range(len(lat_count)), lat_count["count"],
       color=PALETTE[1], edgecolor="#0d1117", alpha=0.8)
ax.set_title("Station Count by Latitude Band")
ax.set_xlabel("Latitude Band"); ax.set_ylabel("Count")
ax.grid(True, alpha=0.3, axis="y")

ax = axes[1]
ax.scatter(df["dist_to_city_km"], df["usage_per_day"],
           c=df["performance_score"], cmap="RdYlGn",
           s=6, alpha=0.35)
ax.set_title("Distance to City vs Usage (colour=performance)")
ax.set_xlabel("Distance to City (km)"); ax.set_ylabel("Users / Day")
ax.grid(True, alpha=0.3)

plt.tight_layout()
save(fig, "13_geospatial_density.png")


# ═════════════════════════════════════════════════════════════════════════════
# 8. RELATIONSHIP DISCOVERY
# ═════════════════════════════════════════════════════════════════════════════
section("8 · RELATIONSHIP DISCOVERY")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Feature Relationship Insights")

# 8.1 usage vs dist_to_city split by charger type
ax = axes[0, 0]
for k, ct in enumerate(df["charger_type"].unique()):
    sub = df[df["charger_type"] == ct]
    ax.scatter(sub["dist_to_city_km"], sub["usage_per_day"],
               s=7, alpha=0.3, color=PALETTE[k], label=ct)
ax.set_title("Distance to City vs Usage")
ax.set_xlabel("Distance (km)"); ax.set_ylabel("Users/Day")
ax.legend(markerscale=2, fontsize=7); ax.grid(True, alpha=0.3)

# 8.2 capacity vs usage colored by is_fast
ax = axes[0, 1]
ax.scatter(df["capacity_kw"], df["usage_per_day"],
           c=df["is_fast_charger"],
           cmap="coolwarm", s=6, alpha=0.35)
ax.set_title("Capacity vs Usage (Blue=DC Fast)")
ax.set_xlabel("Capacity (kW)"); ax.set_ylabel("Users/Day")
ax.grid(True, alpha=0.3)

# 8.3 rating vs performance
ax = axes[0, 2]
ax.hexbin(df["rating"], df["performance_score"],
          gridsize=25, cmap="YlGn")
ax.set_title("Rating vs Performance Score (hexbin)")
ax.set_xlabel("Rating"); ax.set_ylabel("Performance Score")

# 8.4 age vs cost_efficiency
ax = axes[1, 0]
ax.scatter(df["station_age_yrs"], df["cost_efficiency"],
           c=df["charger_type_num"], cmap="plasma",
           s=6, alpha=0.35)
ax.set_title("Station Age vs Cost Efficiency")
ax.set_xlabel("Age (years)"); ax.set_ylabel("Cost Efficiency")
ax.grid(True, alpha=0.3)

# 8.5 parking spots vs revenue proxy
ax = axes[1, 1]
park_rev = df.groupby("parking_spots")["revenue_proxy"].mean()
ax.bar(park_rev.index, park_rev.values,
       color=PALETTE[4], edgecolor="#0d1117")
ax.set_title("Parking Spots vs Avg Monthly Revenue")
ax.set_xlabel("Parking Spots"); ax.set_ylabel("Avg Revenue (USD)")
ax.grid(True, alpha=0.3, axis="y")

# 8.6 renewable + availability vs usage
ax = axes[1, 2]
pivot = df.pivot_table(values="usage_per_day",
                       index="renewable",
                       columns="availability",
                       aggfunc="mean")
sns.heatmap(pivot, annot=True, fmt=".1f",
            cmap="RdYlGn", ax=ax,
            linewidths=0.4, linecolor="#1e2d4a",
            annot_kws={"size": 10})
ax.set_title("Avg Usage: Renewable × Availability")

plt.tight_layout()
save(fig, "14_relationship_discovery.png")

# Print key insights
print("\n  📊 Key Relationship Insights:")
dc_fast = df[df["charger_type"] == "DC Fast Charger"]["usage_per_day"].mean()
ac2     = df[df["charger_type"] == "AC Level 2"]["usage_per_day"].mean()
ac1     = df[df["charger_type"] == "AC Level 1"]["usage_per_day"].mean()
print(f"    • DC Fast avg usage : {dc_fast:.1f} | AC Level 2 : {ac2:.1f} | AC Level 1 : {ac1:.1f}")

best_op = df.groupby("operator")["usage_per_day"].mean().idxmax()
worst_op= df.groupby("operator")["usage_per_day"].mean().idxmin()
print(f"    • Best operator  : {best_op}  ({df[df['operator']==best_op]['usage_per_day'].mean():.1f} avg)")
print(f"    • Worst operator : {worst_op} ({df[df['operator']==worst_op]['usage_per_day'].mean():.1f} avg)")

urban_u = df[df["proximity_category"] == "Urban"]["usage_per_day"].mean()
rural_u = df[df["proximity_category"] == "Rural"]["usage_per_day"].mean()
print(f"    • Urban avg usage : {urban_u:.1f} vs Rural avg : {rural_u:.1f}")


# ═════════════════════════════════════════════════════════════════════════════
# 9. ANOMALY DETECTION
# ═════════════════════════════════════════════════════════════════════════════
section("9 · ANOMALY DETECTION")

# ── 9.1  IQR-based ───────────────────────────────────────────────────────────
Q1  = df["usage_per_day"].quantile(0.25)
Q3  = df["usage_per_day"].quantile(0.75)
IQR = Q3 - Q1
lo_iqr = Q1 - 1.5 * IQR
hi_iqr = Q3 + 1.5 * IQR
df["iqr_anomaly"] = ((df["usage_per_day"] < lo_iqr) |
                     (df["usage_per_day"] > hi_iqr)).astype(int)

# ── 9.2  Z-score ─────────────────────────────────────────────────────────────
z_scores = np.abs(stats.zscore(df["usage_per_day"]))
df["z_score"]       = z_scores
df["zscore_anomaly"] = (z_scores > 2.5).astype(int)

# ── 9.3  Cost anomaly ─────────────────────────────────────────────────────────
df["high_cost_low_use"] = ((df["cost_per_kwh"] > 0.40) &
                           (df["usage_per_day"] < 30)).astype(int)

# ── 9.4  Isolation Forest (multi-feature) ────────────────────────────────────
iso_feats = ["usage_per_day", "cost_per_kwh", "capacity_kw",
             "rating", "dist_to_city_km", "performance_score"]
iso_X = df[iso_feats].copy()
iso_sc = StandardScaler()
iso_Xs = iso_sc.fit_transform(iso_X)

iso_forest = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
df["iso_forest_flag"] = (iso_forest.fit_predict(iso_Xs) == -1).astype(int)

# ── 9.5  Combined anomaly flag ────────────────────────────────────────────────
df["is_anomaly"] = (
    (df["iqr_anomaly"]     == 1) |
    (df["zscore_anomaly"]  == 1) |
    (df["high_cost_low_use"] == 1) |
    (df["iso_forest_flag"] == 1)
).astype(int)

df["anomaly_type"] = "Normal"
df.loc[df["usage_per_day"] > hi_iqr, "anomaly_type"] = "Extreme High Usage"
df.loc[df["usage_per_day"] < lo_iqr, "anomaly_type"] = "Extreme Low Usage"
df.loc[df["high_cost_low_use"] == 1,  "anomaly_type"] = "High Cost / Low Demand"
df.loc[(df["iso_forest_flag"] == 1) &
       (df["anomaly_type"] == "Normal"), "anomaly_type"] = "Isolation Forest Flag"

n_anom = df["is_anomaly"].sum()
print(f"\n  Total anomalies detected : {n_anom} ({n_anom/len(df)*100:.1f}%)")
print(f"\n  Anomaly type breakdown:")
print(df[df["is_anomaly"]==1]["anomaly_type"].value_counts().to_string())

# ── 9.6  Anomaly visualisations ───────────────────────────────────────────────
anom_color_map = {
    "Normal":                   PALETTE[7] + "55",
    "Extreme High Usage":       PALETTE[2],
    "Extreme Low Usage":        PALETTE[1],
    "High Cost / Low Demand":   PALETTE[6],
    "Isolation Forest Flag":    PALETTE[3],
}

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("Anomaly Detection Results")

ax = axes[0, 0]
normal  = df[df["is_anomaly"] == 0]
anomals = df[df["is_anomaly"] == 1]
ax.scatter(normal["cost_per_kwh"],  normal["usage_per_day"],
           s=5, alpha=0.2, color=PALETTE[7], label="Normal")
for at, col in anom_color_map.items():
    if at == "Normal": continue
    sub = anomals[anomals["anomaly_type"] == at]
    if len(sub):
        ax.scatter(sub["cost_per_kwh"], sub["usage_per_day"],
                   s=20, alpha=0.8, color=col, label=at, zorder=4)
ax.axhline(hi_iqr, color=PALETTE[2], ls="--", lw=1, label=f"IQR upper={hi_iqr:.0f}")
ax.axhline(lo_iqr, color=PALETTE[1], ls="--", lw=1, label=f"IQR lower={lo_iqr:.0f}")
ax.set_title("Cost vs Usage — Anomalies Highlighted")
ax.set_xlabel("Cost (USD/kWh)"); ax.set_ylabel("Users / Day")
ax.legend(fontsize=7, markerscale=2); ax.grid(True, alpha=0.25)

ax = axes[0, 1]
ax.hist(df["z_score"], bins=40, color=PALETTE[0],
        edgecolor="#0d1117", alpha=0.85)
ax.axvline(2.5, color=PALETTE[6], ls="--", lw=2, label="Z=2.5 threshold")
ax.set_title("Z-Score Distribution")
ax.set_xlabel("Absolute Z-Score"); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1, 0]
at_counts = df[df["is_anomaly"]==1]["anomaly_type"].value_counts()
ax.barh(at_counts.index, at_counts.values,
        color=[anom_color_map.get(k, "#555") for k in at_counts.index])
ax.set_title("Anomaly Type Breakdown")
ax.set_xlabel("Count"); ax.grid(True, alpha=0.3, axis="x")

ax = axes[1, 1]
op_anom = df.groupby("operator")["is_anomaly"].sum().sort_values()
ax.barh(op_anom.index, op_anom.values,
        color=PALETTE[:len(op_anom)], edgecolor="#0d1117")
ax.set_title("Anomaly Count by Operator")
ax.set_xlabel("Anomaly Count"); ax.grid(True, alpha=0.3, axis="x")

plt.tight_layout()
save(fig, "15_anomaly_detection.png")

# Print top anomalous stations
print("\n  Top 10 anomalous stations:")
print(df[df["is_anomaly"]==1][
    ["station_id","charger_type","operator","usage_per_day",
     "cost_per_kwh","z_score","anomaly_type"]
].sort_values("z_score", ascending=False).head(10).to_string(index=False))


# ═════════════════════════════════════════════════════════════════════════════
# 10. PREDICTIVE MODELING
# ═════════════════════════════════════════════════════════════════════════════
section("10 · PREDICTIVE MODELING")

# ── 10.1  Prepare feature matrix ─────────────────────────────────────────────
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

# Ordinal encode 'era' as extra feature
era_enc = OrdinalEncoder(categories=[["Early","Growth","Mature","Recent"]])
X["era_num"] = era_enc.fit_transform(df[["era"]].astype(str))

# Scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)

sc_model = StandardScaler()
X_tr_sc  = sc_model.fit_transform(X_train)
X_te_sc  = sc_model.transform(X_test)

print(f"\n  Training samples : {len(X_train):,}")
print(f"  Testing  samples : {len(X_test):,}")
print(f"  Features used    : {X.shape[1]}")

# ── 10.2  Models ─────────────────────────────────────────────────────────────
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression":  Ridge(alpha=1.0),
    "Random Forest":     RandomForestRegressor(
                            n_estimators=150, max_depth=12,
                            random_state=42, n_jobs=-1)
}

results = {}
for mname, model in models.items():
    use_scaled = mname != "Random Forest"
    Xtr = X_tr_sc if use_scaled else X_train.values
    Xte = X_te_sc if use_scaled else X_test.values

    model.fit(Xtr, y_train)
    pred = model.predict(Xte)
    cv   = cross_val_score(model, Xtr, y_train, cv=5,
                           scoring="r2", n_jobs=-1)

    mae  = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2   = r2_score(y_test, pred)

    results[mname] = {"model": model, "pred": pred,
                      "MAE": mae, "RMSE": rmse, "R2": r2,
                      "CV_R2_mean": cv.mean(), "CV_R2_std": cv.std()}
    print(f"\n  ── {mname}")
    print(f"     MAE  = {mae:.3f}")
    print(f"     RMSE = {rmse:.3f}")
    print(f"     R²   = {r2:.4f}")
    print(f"     CV R² (5-fold) = {cv.mean():.4f} ± {cv.std():.4f}")

# ── 10.3  Feature importance (Random Forest) ─────────────────────────────────
rf_model  = results["Random Forest"]["model"]
feat_imp  = pd.Series(rf_model.feature_importances_,
                      index=X.columns).sort_values(ascending=False)

print(f"\n  Top 15 feature importances (Random Forest):")
print(feat_imp.head(15).round(4).to_string())

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("Feature Importance — Random Forest")

ax = axes[0]
top15 = feat_imp.head(15)
ax.barh(top15.index[::-1], top15.values[::-1],
        color=PALETTE[0], edgecolor="#0d1117")
ax.set_title("Top 15 Feature Importances")
ax.set_xlabel("Importance"); ax.grid(True, alpha=0.3, axis="x")

ax = axes[1]
cumulative = feat_imp.values.cumsum() / feat_imp.values.sum() * 100
ax.plot(range(1, len(cumulative)+1), cumulative,
        color=PALETTE[2], lw=2, marker="o", ms=3)
ax.axhline(80, color=PALETTE[6], ls="--", lw=1.5, label="80% threshold")
ax.axhline(95, color=PALETTE[4], ls="--", lw=1.5, label="95% threshold")
ax.set_title("Cumulative Feature Importance")
ax.set_xlabel("Number of Features"); ax.set_ylabel("Cumulative Importance (%)")
ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
save(fig, "16_feature_importance.png")

# ── 10.4  Model comparison + Actual vs Predicted ─────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Predictive Model Evaluation")

metrics_df = pd.DataFrame({
    m: {"MAE": v["MAE"], "RMSE": v["RMSE"], "R²": v["R2"]}
    for m, v in results.items()
}).T

for ci, metric in enumerate(["MAE", "RMSE", "R²"]):
    ax = axes[0, ci]
    colors = [PALETTE[0], PALETTE[2], PALETTE[5]]
    bars = ax.bar(metrics_df.index, metrics_df[metric],
                  color=colors, edgecolor="#0d1117")
    ax.set_title(f"Model Comparison — {metric}")
    ax.set_xticklabels(metrics_df.index, rotation=12, ha="right")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, metrics_df[metric]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

for ri, (mname, mcolor) in enumerate(zip(
        ["Linear Regression","Ridge Regression","Random Forest"],
        [PALETTE[0], PALETTE[2], PALETTE[5]])):
    ax = axes[1, ri]
    pred = results[mname]["pred"]
    ax.scatter(y_test, pred, s=5, alpha=0.3, color=mcolor)
    perfect = np.linspace(y_test.min(), y_test.max(), 100)
    ax.plot(perfect, perfect, color="white", lw=1.5, ls="--", label="Perfect")
    ax.set_title(f"{mname}\nR²={results[mname]['R2']:.4f}  MAE={results[mname]['MAE']:.2f}")
    ax.set_xlabel("Actual Usage"); ax.set_ylabel("Predicted Usage")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

plt.tight_layout()
save(fig, "17_model_evaluation.png")

# Residuals plot
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Residuals Analysis")
for ax, (mname, mcolor) in zip(axes, [("Linear Regression",PALETTE[0]),
                                       ("Ridge Regression",PALETTE[2]),
                                       ("Random Forest",PALETTE[5])]):
    residuals = y_test.values - results[mname]["pred"]
    ax.scatter(results[mname]["pred"], residuals,
               s=5, alpha=0.3, color=mcolor)
    ax.axhline(0, color="white", lw=1.5, ls="--")
    ax.set_title(f"Residuals — {mname}")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Residual")
    ax.grid(True, alpha=0.25)
plt.tight_layout()
save(fig, "18_residuals.png")


# ═════════════════════════════════════════════════════════════════════════════
# 11. COMPREHENSIVE SUMMARY DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
section("11 · COMPREHENSIVE SUMMARY DASHBOARD")

fig = plt.figure(figsize=(22, 16))
fig.suptitle("EV Charging Station Analytics — Executive Summary",
             fontsize=17, y=0.99, fontweight="bold")

gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.35)

# ── Usage by charger type ────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ct_usage = df.groupby("charger_type")["usage_per_day"].mean().sort_values()
ax1.barh(ct_usage.index, ct_usage.values,
         color=[CHARGER_COLOR := [PALETTE[1],PALETTE[0],PALETTE[2]]][0])
ax1.set_title("Avg Usage by Charger"); ax1.grid(True, alpha=0.3, axis="x")

# ── Usage by operator ────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
op_u = df.groupby("operator")["usage_per_day"].mean().sort_values()
ax2.barh(op_u.index, op_u.values, color=PALETTE[:len(op_u)])
ax2.set_title("Avg Usage by Operator"); ax2.grid(True, alpha=0.3, axis="x")

# ── Demand pie ───────────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
dm = df["demand_level"].value_counts()
ax3.pie(dm.values, labels=dm.index,
        colors=[PALETTE[2],PALETTE[1],PALETTE[5]],
        autopct="%1.1f%%", startangle=90,
        textprops={"color":"#e6edf3","fontsize":9})
ax3.set_title("Demand Level Split")

# ── Performance by proximity ─────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[0, 3])
prox = df.groupby("proximity_category", observed=True)["performance_score"].mean()
ax4.bar(prox.index, prox.values,
        color=[PALETTE[0],PALETTE[3],PALETTE[4]], edgecolor="#0d1117")
ax4.set_title("Performance by Proximity"); ax4.grid(True, alpha=0.3, axis="y")

# ── Cluster scatter PCA ──────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, :2])
for ci, (cname, ccolor) in enumerate(zip(CLUSTER_NAMES.values(), ccolors)):
    m = df["cluster"] == ci
    ax5.scatter(df.loc[m,"pca1"], df.loc[m,"pca2"],
                s=5, alpha=0.3, color=ccolor, label=cname)
ax5.set_title(f"K-Means Clusters (PCA 2D, k=4, sil={sil:.3f})")
ax5.set_xlabel("PC1"); ax5.set_ylabel("PC2")
ax5.legend(fontsize=7, markerscale=3); ax5.grid(True, alpha=0.2)

# ── Anomaly scatter ──────────────────────────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2:])
ax6.scatter(df[df["is_anomaly"]==0]["cost_per_kwh"],
            df[df["is_anomaly"]==0]["usage_per_day"],
            s=4, alpha=0.15, color=PALETTE[7], label="Normal")
for at, col in anom_color_map.items():
    if at == "Normal": continue
    sub = df[df["anomaly_type"] == at]
    if len(sub):
        ax6.scatter(sub["cost_per_kwh"], sub["usage_per_day"],
                    s=18, alpha=0.85, color=col, label=f"{at} (n={len(sub)})", zorder=4)
ax6.set_title(f"Anomaly Map (n={n_anom} flagged)")
ax6.set_xlabel("Cost (USD/kWh)"); ax6.set_ylabel("Users / Day")
ax6.legend(fontsize=7, markerscale=2); ax6.grid(True, alpha=0.25)

# ── Feature importance ───────────────────────────────────────────────────────
ax7 = fig.add_subplot(gs[2, :2])
top10 = feat_imp.head(10)
ax7.barh(top10.index[::-1], top10.values[::-1],
         color=PALETTE[0], edgecolor="#0d1117")
ax7.set_title(f"Top 10 Feature Importances (RF R²={results['Random Forest']['R2']:.4f})")
ax7.set_xlabel("Importance"); ax7.grid(True, alpha=0.3, axis="x")

# ── Year trend ───────────────────────────────────────────────────────────────
ax8 = fig.add_subplot(gs[2, 2:])
yr = df.groupby("install_year").agg(
    new_stations=("station_id","count"),
    avg_usage=("usage_per_day","mean")
).reset_index()
ax8b = ax8.twinx()
ax8.bar(yr["install_year"], yr["new_stations"],
        color=f"{PALETTE[1]}55", edgecolor="#0d1117", label="New Stations")
ax8b.plot(yr["install_year"], yr["avg_usage"],
          color=PALETTE[0], marker="o", ms=4, lw=2, label="Avg Usage")
ax8.set_title("YoY Station Deployment & Avg Usage")
ax8.set_xlabel("Year"); ax8.set_ylabel("Station Count")
ax8b.set_ylabel("Avg Users/Day")
lines, labels = ax8.get_legend_handles_labels()
lines2, labels2 = ax8b.get_legend_handles_labels()
ax8.legend(lines+lines2, labels+labels2, fontsize=8)
ax8.grid(True, alpha=0.2)

plt.tight_layout()
save(fig, "00_executive_summary_dashboard.png")


# ═════════════════════════════════════════════════════════════════════════════
# 12. FINAL REPORT
# ═════════════════════════════════════════════════════════════════════════════
section("12 · FINAL REPORT & INSIGHTS")

print(f"""
  ┌─────────────────────────────────────────────────────────────┐
  │              EV ANALYTICS — FINAL SUMMARY                   │
  ├─────────────────────────────────────────────────────────────┤
  │  DATASET                                                    │
  │   • Total stations    : {len(df):,}                             │
  │   • Original features : {df_raw.shape[1]}                              │
  │   • Engineered feats  : {len(new_feats)} (38 total)                    │
  ├─────────────────────────────────────────────────────────────┤
  │  USAGE STATISTICS                                           │
  │   • Avg daily usage   : {df['usage_per_day'].mean():.1f} users/day             │
  │   • Highest usage type: DC Fast Charger ({dc_fast:.1f} avg)        │
  │   • Best operator     : {best_op:<15}                        │
  │   • High demand (≥70) : {(df['demand_level']=='High').sum():,} stations ({(df['demand_level']=='High').mean()*100:.1f}%)       │
  ├─────────────────────────────────────────────────────────────┤
  │  CLUSTERING (K-Means, k=4)                                  │
  │   • Silhouette Score  : {sil:.4f}                            │
  │   • Largest cluster   : {df['cluster_name'].value_counts().index[0]:<25}     │
  ├─────────────────────────────────────────────────────────────┤
  │  ANOMALIES                                                  │
  │   • Total anomalies   : {n_anom} ({n_anom/len(df)*100:.1f}%)                     │
  │   • IQR bounds        : [{lo_iqr:.1f}, {hi_iqr:.1f}] users/day            │
  ├─────────────────────────────────────────────────────────────┤
  │  BEST PREDICTION MODEL : Random Forest                      │
  │   • R²   = {results['Random Forest']['R2']:.4f}                              │
  │   • MAE  = {results['Random Forest']['MAE']:.3f} users/day                    │
  │   • RMSE = {results['Random Forest']['RMSE']:.3f} users/day                    │
  │   • CV R²= {results['Random Forest']['CV_R2_mean']:.4f} ± {results['Random Forest']['CV_R2_std']:.4f}                    │
  ├─────────────────────────────────────────────────────────────┤
  │  KEY INSIGHTS                                               │
  │   1. DC Fast Chargers have {(dc_fast/ac1 - 1)*100:.0f}% more usage than Level 1  │
  │   2. Urban stations outperform Rural by ~{urban_u-rural_u:.1f} users/day      │
  │   3. 24/7 stations earn {df[df['is_247']==1]['revenue_proxy'].mean() - df[df['is_247']==0]['revenue_proxy'].mean():.0f}% higher monthly revenue       │
  │   4. Renewable stations rate {df[df['is_renewable']==1]['rating'].mean():.2f} vs {df[df['is_renewable']==0]['rating'].mean():.2f} (non-renewable)    │
  │   5. Monthly maintenance drives {df[df['maintenance_freq']=='Monthly']['usage_per_day'].mean() - df[df['maintenance_freq']=='Annually']['usage_per_day'].mean():.1f} more users/day        │
  └─────────────────────────────────────────────────────────────┘
""")

print(f"  ✔ All {len(os.listdir(OUT_DIR))} plots saved to  ./{OUT_DIR}/")
print(f"  ✔ Feature CSV saved to  ./ev_features_engineered.csv")
print("\n  ════════════════════════════════════════════════════")
print("  ANALYSIS COMPLETE")
print("  ════════════════════════════════════════════════════\n")
