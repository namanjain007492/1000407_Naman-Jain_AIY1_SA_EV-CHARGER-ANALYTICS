import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1. PAGE CONFIGURATION & THEME
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="EV Station Analytics", page_icon="⚡", layout="wide")

plt.rcParams.update({
    "figure.facecolor": "#0d1117", "axes.facecolor": "#0d1117",
    "axes.edgecolor": "#30363d", "axes.labelcolor": "#8b949e",
    "axes.titlecolor": "#e6edf3", "xtick.color": "#8b949e",
    "ytick.color": "#8b949e", "grid.color": "#21262d",
    "text.color": "#e6edf3", "font.family": "sans-serif"
})
PALETTE = ["#00e5a0", "#22d3ee", "#f97316", "#a78bfa", "#f472b6", "#fbbf24", "#ef4444", "#60a5fa"]

# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA PROCESSING & 20 ENGINEERED FEATURES (Cached for Performance)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_and_engineer_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "detailed_ev_charging_stations.csv")
    
    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset missing! Please place 'detailed_ev_charging_stations.csv' in {BASE_DIR}")
        st.stop()

    df_raw = pd.read_csv(DATA_PATH)
    df = df_raw.copy()
    
    # Standardize column names
    df.columns = [
        "station_id", "lat", "lng", "address", "charger_type", "cost_per_kwh", 
        "availability", "dist_to_city_km", "usage_per_day", "operator", "capacity_kw", 
        "connector_types", "install_year", "renewable", "rating", "parking_spots", "maintenance_freq"
    ]
    
    # Fill missing values
    for c in df.select_dtypes(include=np.number).columns: 
        df[c].fillna(df[c].median(), inplace=True)
    for c in df.select_dtypes(include="object").columns:  
        df[c].fillna(df[c].mode()[0], inplace=True)
    df.drop_duplicates(subset="station_id", inplace=True)

    # ── ENGINEER 20 NEW FEATURES ──────────────────────────────────────────────
    df["F1_station_age_yrs"] = 2024 - df["install_year"]
    df["F2_usage_per_capacity"] = df["usage_per_day"] / df["capacity_kw"].clip(1)
    df["F3_cost_efficiency"] = (df["usage_per_day"] / df["cost_per_kwh"].clip(0.01)).round(2)
    df["F4_revenue_proxy"] = (df["usage_per_day"] * df["cost_per_kwh"] * 30).round(2)
    df["F5_revenue_per_spot"] = (df["F4_revenue_proxy"] / df["parking_spots"].clip(1)).round(2)
    df["F6_usage_per_spot"] = (df["usage_per_day"] / df["parking_spots"].clip(1)).round(2)
    df["F7_demand_level"] = pd.cut(df["usage_per_day"], bins=[-1, 30, 60, 100], labels=["Low", "Medium", "High"])
    df["F8_performance_score"] = ((df["usage_per_day"]/100)*0.35 + (1-df["cost_per_kwh"]/0.5)*0.20 + 
                                  (df["rating"]/5)*0.25 + (df["capacity_kw"]/350)*0.20).round(4)
    df["F9_proximity_cat"] = pd.cut(df["dist_to_city_km"], bins=[-1, 5, 12, 20], labels=["Urban","Suburban","Rural"])
    df["F10_is_247"] = (df["availability"] == "24/7").astype(int)
    df["F11_is_renewable"] = (df["renewable"] == "Yes").astype(int)
    
    maint_map = {"Monthly": 3, "Quarterly": 2, "Annually": 1}
    df["F12_maintenance_score"] = df["maintenance_freq"].map(maint_map)
    df["F13_num_connectors"] = df["connector_types"].str.split(",").apply(len)
    
    charger_map = {"AC Level 1": 1, "AC Level 2": 2, "DC Fast Charger": 3}
    df["F14_charger_type_num"] = df["charger_type"].map(charger_map)
    df["F15_is_fast_charger"] = (df["F14_charger_type_num"] == 3).astype(int)
    df["F16_capacity_x_247"] = df["capacity_kw"] * df["F10_is_247"]
    df["F17_cost_x_dist"] = df["cost_per_kwh"] * df["dist_to_city_km"]

    # F18: Clustering (K-Means)
    clust_feats = ["usage_per_day", "cost_per_kwh", "capacity_kw", "rating", "dist_to_city_km", "F8_performance_score"]
    X_sc = StandardScaler().fit_transform(df[clust_feats])
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["F18_cluster"] = km.fit_predict(X_sc)
    
    # For PCA Visualization later
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_sc)
    df["pca1"], df["pca2"] = pca_coords[:, 0], pca_coords[:, 1]

    # F19 & F20: Anomaly Detection
    iso = IsolationForest(contamination=0.05, random_state=42)
    df["F19_is_anomaly"] = (iso.fit_predict(X_sc) == -1).astype(int)
    
    # Assign Anomaly Reason
    df["F20_anomaly_reason"] = "Normal"
    df.loc[(df["F19_is_anomaly"] == 1) & (df["usage_per_day"] > 80), "F20_anomaly_reason"] = "Extreme High Usage"
    df.loc[(df["F19_is_anomaly"] == 1) & (df["cost_per_kwh"] > 0.40) & (df["usage_per_day"] < 20), "F20_anomaly_reason"] = "High Cost/Low Demand"
    df.loc[(df["F19_is_anomaly"] == 1) & (df["F20_anomaly_reason"] == "Normal"), "F20_anomaly_reason"] = "Complex Outlier"

    return df

df = load_and_engineer_data()

# ─────────────────────────────────────────────────────────────────────────────
# 3. SIDEBAR CONTROLS & DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("⚡ Settings & Filters")
st.sidebar.markdown("Filter the dashboard dynamically.")

op_filter = st.sidebar.multiselect("Operator", df["operator"].unique(), default=df["operator"].unique()[:3])
ct_filter = st.sidebar.multiselect("Charger Type", df["charger_type"].unique(), default=df["charger_type"].unique())

filtered_df = df[(df["operator"].isin(op_filter)) & (df["charger_type"].isin(ct_filter))]

st.sidebar.divider()
st.sidebar.subheader("Export Data")
csv = df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="📥 Download Processed CSV", 
    data=csv, 
    file_name="ev_engineered_data.csv", 
    mime="text/csv"
)

# ─────────────────────────────────────────────────────────────────────────────
# 4. MAIN DASHBOARD LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
st.title("EV Charging Analytics Engine")
st.markdown("Exploring global charging stations using 20 engineered features, clustering, and predictive modeling.")

# 5 Sleek Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Executive Summary", 
    "📈 Exploratory Data", 
    "🌍 Geospatial", 
    "🚨 Anomalies", 
    "🤖 Predictive Model"
])

# ── TAB 1: EXECUTIVE SUMMARY ─────────────────────────────────────────────────
with tab1:
    st.subheader("Global KPI Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active Stations", f"{len(filtered_df):,}")
    c2.metric("Avg Daily Usage", f"{filtered_df['usage_per_day'].mean():.1f} users")
    c3.metric("Avg Cost per kWh", f"${filtered_df['cost_per_kwh'].mean():.2f}")
    c4.metric("Avg Performance Score", f"{filtered_df['F8_performance_score'].mean():.2f}/1.0")
    
    st.divider()
    
    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Estimated Monthly Revenue by Operator**")
        fig, ax = plt.subplots(figsize=(8, 4))
        op_rev = filtered_df.groupby("operator")["F4_revenue_proxy"].mean().sort_values(ascending=False)
        ax.bar(op_rev.index, op_rev.values, color=PALETTE[:len(op_rev)])
        ax.set_ylabel("Monthly Revenue (USD)")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)
        
    with colB:
        st.markdown("**Demand Level Distribution**")
        fig, ax = plt.subplots(figsize=(8, 4))
        dm = filtered_df["F7_demand_level"].value_counts()
        ax.pie(dm.values, labels=dm.index, colors=[PALETTE[2], PALETTE[1], PALETTE[5]], autopct="%1.1f%%", textprops={"color":"#e6edf3"})
        st.pyplot(fig)

# ── TAB 2: EXPLORATORY DATA ──────────────────────────────────────────────────
with tab2:
    st.subheader("Deep Dive Analytics")
    colA, colB = st.columns(2)
    
    with colA:
        st.markdown("**Cost Efficiency vs Age**")
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(data=filtered_df, x="F1_station_age_yrs", y="F3_cost_efficiency", 
                        hue="charger_type", palette=PALETTE[:3], alpha=0.6, ax=ax)
        st.pyplot(fig)
        
    with colB:
        st.markdown("**Performance by Proximity to City**")
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.boxplot(data=filtered_df, x="F9_proximity_cat", y="F8_performance_score", palette=PALETTE[:3], ax=ax)
        st.pyplot(fig)

# ── TAB 3: GEOSPATIAL ────────────────────────────────────────────────────────
with tab3:
    st.subheader("Geospatial Footprint")
    st.markdown("Filter coordinates based on your sidebar selections. This map is natively interactive (Zoom/Pan).")
    st.map(filtered_df[["lat", "lng"]], color="#00e5a0", zoom=1)

# ── TAB 4: ANOMALIES & CLUSTERING ────────────────────────────────────────────
with tab4:
    colA, colB = st.columns(2)
    
    with colA:
        st.subheader("Anomaly Detection (Isolation Forest)")
        st.markdown(f"Detected **{filtered_df['F19_is_anomaly'].sum()}** anomalies in the current view.")
        
        fig, ax = plt.subplots(figsize=(7, 5))
        normal = filtered_df[filtered_df["F19_is_anomaly"] == 0]
        anomalies = filtered_df[filtered_df["F19_is_anomaly"] == 1]
        
        ax.scatter(normal["cost_per_kwh"], normal["usage_per_day"], color=PALETTE[7], alpha=0.2, label="Normal")
        ax.scatter(anomalies["cost_per_kwh"], anomalies["usage_per_day"], color=PALETTE[6], alpha=0.9, s=40, label="Anomaly")
        ax.set_xlabel("Cost (USD/kWh)")
        ax.set_ylabel("Usage / Day")
        ax.legend()
        st.pyplot(fig)

    with colB:
        st.subheader("K-Means Clustering (k=4)")
        st.markdown("PCA-reduced visualization of station operational clusters.")
        fig, ax = plt.subplots(figsize=(7, 5))
        for ci in range(4):
            mask = filtered_df["F18_cluster"] == ci
            ax.scatter(filtered_df.loc[mask, "pca1"], filtered_df.loc[mask, "pca2"], 
                       s=15, alpha=0.6, label=f"Cluster {ci}", color=PALETTE[ci])
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend()
        st.pyplot(fig)

# ── TAB 5: PREDICTIVE MODELING ───────────────────────────────────────────────
with tab5:
    st.subheader("Predictive Analytics: Estimating Daily Usage")
    
    # Train model automatically on the full dataset using our new engineered features
    features = ["cost_per_kwh", "capacity_kw", "dist_to_city_km", "rating", 
                "F1_station_age_yrs", "F15_is_fast_charger", "F10_is_247", "F12_maintenance_score"]
    
    X = df[features]
    y = df["usage_per_day"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Model R² Score", f"{r2_score(y_test, preds):.3f}")
    c2.metric("Mean Absolute Error", f"{mean_absolute_error(y_test, preds):.2f}")
    c3.metric("Root Mean Squared Error", f"{np.sqrt(mean_squared_error(y_test, preds)):.2f}")
    
    st.divider()
    
    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Feature Importance (Random Forest)**")
        fig, ax = plt.subplots(figsize=(7, 4))
        importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=True)
        ax.barh(importances.index, importances.values, color=PALETTE[0])
        ax.set_xlabel("Relative Importance")
        st.pyplot(fig)
        
    with colB:
        st.markdown("**Actual vs Predicted Results**")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(y_test, preds, alpha=0.4, color=PALETTE[1], s=15)
        perfect = np.linspace(y_test.min(), y_test.max(), 100)
        ax.plot(perfect, perfect, color="white", lw=2, ls="--", label="Perfect Accuracy")
        ax.set_xlabel("Actual Usage")
        ax.set_ylabel("Predicted Usage")
        ax.legend()
        st.pyplot(fig)
