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
# 2. DATA PROCESSING & 22 ENGINEERED FEATURES
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
    
    # FIX: Standardize column names (changed 'lng' to 'lon' for Streamlit compatibility)
    df.columns = [
        "station_id", "lat", "lon", "address", "charger_type", "cost_per_kwh", 
        "availability", "dist_to_city_km", "usage_per_day", "operator", "capacity_kw", 
        "connector_types", "install_year", "renewable", "rating", "parking_spots", "maintenance_freq"
    ]
    
    for c in df.select_dtypes(include=np.number).columns: df[c].fillna(df[c].median(), inplace=True)
    for c in df.select_dtypes(include="object").columns:  df[c].fillna(df[c].mode()[0], inplace=True)
    df.drop_duplicates(subset="station_id", inplace=True)

    # ── ENGINEER 22 NEW FEATURES ──────────────────────────────────────────────
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
    
    # NEW: Environmental and Business Metrics
    df["F18_co2_saved_kg_month"] = (df["usage_per_day"] * 7.5 * 30).round(1) # Est 7.5kg saved per charge
    df["F19_est_roi_years"] = ((df["capacity_kw"] * 800) / (df["F4_revenue_proxy"] * 12 + 1)).round(1)

    # F20: Clustering (K-Means)
    clust_feats = ["usage_per_day", "cost_per_kwh", "capacity_kw", "rating", "dist_to_city_km", "F8_performance_score"]
    X_sc = StandardScaler().fit_transform(df[clust_feats])
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["F20_cluster"] = km.fit_predict(X_sc)
    
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_sc)
    df["pca1"], df["pca2"] = pca_coords[:, 0], pca_coords[:, 1]

    # F21 & F22: Anomaly Detection
    iso = IsolationForest(contamination=0.05, random_state=42)
    df["F21_is_anomaly"] = (iso.fit_predict(X_sc) == -1).astype(int)
    
    df["F22_anomaly_reason"] = "Normal"
    df.loc[(df["F21_is_anomaly"] == 1) & (df["usage_per_day"] > 80), "F22_anomaly_reason"] = "Extreme High Usage"
    df.loc[(df["F21_is_anomaly"] == 1) & (df["cost_per_kwh"] > 0.40) & (df["usage_per_day"] < 20), "F22_anomaly_reason"] = "High Cost/Low Demand"
    df.loc[(df["F21_is_anomaly"] == 1) & (df["F22_anomaly_reason"] == "Normal"), "F22_anomaly_reason"] = "Complex Outlier"

    return df

df = load_and_engineer_data()

# ─────────────────────────────────────────────────────────────────────────────
# 3. SIDEBAR CONTROLS
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("⚡ Settings & Filters")
op_filter = st.sidebar.multiselect("Operator", df["operator"].unique(), default=df["operator"].unique()[:3])
ct_filter = st.sidebar.multiselect("Charger Type", df["charger_type"].unique(), default=df["charger_type"].unique())

filtered_df = df[(df["operator"].isin(op_filter)) & (df["charger_type"].isin(ct_filter))]

st.sidebar.divider()
st.sidebar.subheader("Export Data")
csv = df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button("📥 Download Processed CSV", data=csv, file_name="ev_engineered_data.csv", mime="text/csv")

# ─────────────────────────────────────────────────────────────────────────────
# 4. MAIN DASHBOARD LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
st.title("EV Charging Analytics Engine")
st.markdown("Exploring global charging stations through data engineering, geospatial mapping, and machine learning.")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Summary", "📈 EDA", "🌍 Map", "🚨 Anomalies", "🤖 ML Model", "🎮 EV Fun & Facts"
])

# ── TAB 1: EXECUTIVE SUMMARY ─────────────────────────────────────────────────
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active Stations", f"{len(filtered_df):,}")
    c2.metric("Avg Daily Usage", f"{filtered_df['usage_per_day'].mean():.1f}")
    c3.metric("Total CO₂ Saved / Month", f"{filtered_df['F18_co2_saved_kg_month'].sum() / 1000:,.1f} Tons")
    c4.metric("Avg Performance Score", f"{filtered_df['F8_performance_score'].mean():.2f}/1.0")
    
    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Estimated Monthly Revenue by Operator**")
        fig, ax = plt.subplots(figsize=(8, 4))
        op_rev = filtered_df.groupby("operator")["F4_revenue_proxy"].mean().sort_values(ascending=False)
        ax.bar(op_rev.index, op_rev.values, color=PALETTE[:len(op_rev)])
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
    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Cost Efficiency vs Age**")
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(data=filtered_df, x="F1_station_age_yrs", y="F3_cost_efficiency", hue="charger_type", palette=PALETTE[:3], alpha=0.6, ax=ax)
        st.pyplot(fig)
    with colB:
        st.markdown("**Performance by Proximity to City**")
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.boxplot(data=filtered_df, x="F9_proximity_cat", y="F8_performance_score", palette=PALETTE[:3], ax=ax)
        st.pyplot(fig)

# ── TAB 3: GEOSPATIAL ────────────────────────────────────────────────────────
with tab3:
    st.subheader("Geospatial Footprint")
    st.markdown("This map natively reads the `lat` and `lon` columns. Zoom and pan to explore.")
    st.map(filtered_df[["lat", "lon"]], color="#00e5a0", zoom=1)

# ── TAB 4: ANOMALIES & CLUSTERING ────────────────────────────────────────────
with tab4:
    colA, colB = st.columns(2)
    with colA:
        st.subheader("Anomaly Detection")
        st.markdown(f"Detected **{filtered_df['F21_is_anomaly'].sum()}** anomalies.")
        fig, ax = plt.subplots(figsize=(7, 5))
        normal = filtered_df[filtered_df["F21_is_anomaly"] == 0]
        anomalies = filtered_df[filtered_df["F21_is_anomaly"] == 1]
        ax.scatter(normal["cost_per_kwh"], normal["usage_per_day"], color=PALETTE[7], alpha=0.2, label="Normal")
        ax.scatter(anomalies["cost_per_kwh"], anomalies["usage_per_day"], color=PALETTE[6], alpha=0.9, s=40, label="Anomaly")
        ax.set_xlabel("Cost (USD/kWh)")
        ax.set_ylabel("Usage / Day")
        ax.legend()
        st.pyplot(fig)
    with colB:
        st.subheader("K-Means Clustering")
        fig, ax = plt.subplots(figsize=(7, 5))
        for ci in range(4):
            mask = filtered_df["F20_cluster"] == ci
            ax.scatter(filtered_df.loc[mask, "pca1"], filtered_df.loc[mask, "pca2"], s=15, alpha=0.6, label=f"Cluster {ci}", color=PALETTE[ci])
        ax.legend()
        st.pyplot(fig)

# ── TAB 5: PREDICTIVE MODELING ───────────────────────────────────────────────
with tab5:
    features = ["cost_per_kwh", "capacity_kw", "dist_to_city_km", "rating", "F1_station_age_yrs", "F15_is_fast_charger", "F10_is_247", "F12_maintenance_score"]
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
    
    fig, ax = plt.subplots(figsize=(10, 3))
    importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=True)
    ax.barh(importances.index, importances.values, color=PALETTE[0])
    st.pyplot(fig)

# ── TAB 6: EV FUN & FACTS ────────────────────────────────────────────────────
with tab6:
    st.header("💡 Did You Know? (EV Trivia)")
    
    st.markdown("""
    * **Massive Global Shift:** The global market for Electric and Fuel Cell Vehicles is projected to hit an incredible **$1.8 Trillion** by 2029!
    * **Extreme Reliability:** Despite common misconceptions, EV batteries are incredibly durable. The battery failure rate for EVs on the road is currently estimated to be as low as **0.003%**.
    * **Car Wash Safe:** Yes, you can take an EV through an automated car wash! Modern EVs utilize advanced 4-stage electric shock prevention designs to keep high-voltage components completely isolated from water.
    * **Plug it in anywhere:** You don't always need a specialized wall box. An EV can technically be charged using a standard 3-prong household outlet (known as Level 1 charging), though it is much slower.
    """)
    
    st.divider()
    
    st.header("🎮 Mini-Game: The EV Trip Estimator")
    st.markdown("Play around with these variables to see how environmental factors impact an EV's range!")
    
    col_game1, col_game2 = st.columns(2)
    with col_game1:
        battery_size = st.slider("Battery Size (kWh)", min_value=30, max_value=120, value=65, step=5)
        efficiency = st.slider("Vehicle Efficiency (km per kWh)", min_value=3.0, max_value=8.0, value=5.5, step=0.5)
        weather = st.selectbox("Driving Condition", ["Ideal City Driving (20°C)", "Freezing Winter (-5°C)", "Highway Cruising (120km/h)"])
    
    with col_game2:
        # Calculate theoretical range
        base_range = battery_size * efficiency
        
        # Apply penalties based on physics
        if weather == "Freezing Winter (-5°C)":
            actual_range = base_range * 0.75  # 25% penalty for heating and battery chemistry
            st.warning("Cold weather slows down the battery's chemical reactions and uses energy for cabin heating.")
        elif weather == "Highway Cruising (120km/h)":
            actual_range = base_range * 0.85  # 15% penalty for aerodynamic drag
            st.info("EVs are highly efficient in the city due to regenerative braking, but aerodynamic drag eats up range at high speeds.")
        else:
            actual_range = base_range
            st.success("Perfect conditions for maximum efficiency!")
            
        st.metric(label="Estimated Real-World Range", value=f"{int(actual_range)} km")
