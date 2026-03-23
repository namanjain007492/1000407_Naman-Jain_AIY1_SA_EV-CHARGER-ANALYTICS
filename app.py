import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import pydeck as pdk

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1. PAGE CONFIGURATION & THEME
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="EV Analytics Pro", page_icon="⚡", layout="wide")

# Set a professional dark theme palette for Plotly/UI
PALETTE = ["#e63946", "#f4a261", "#2a9d8f", "#e9c46a", "#264653", "#8ab17d", "#e76f51", "#1d3557"]

# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA PROCESSING & 38 ENGINEERED FEATURES (Summative Assessment Pipeline)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_and_engineer_data():
    # Robust Pathing
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "detailed_ev_charging_stations.csv")
    
    if not os.path.exists(DATA_PATH):
        st.error("Dataset missing! Please ensure 'detailed_ev_charging_stations.csv' is in the project folder.")
        st.stop()

    df = pd.read_csv(DATA_PATH)
    
    # Standardize column names (Streamlit maps strictly require 'lat' and 'lon')
    df.columns = [
        "station_id", "lat", "lon", "address", "charger_type", "cost_per_kwh", 
        "availability", "dist_to_city_km", "usage_per_day", "operator", "capacity_kw", 
        "connector_types", "install_year", "renewable", "rating", "parking_spots", "maintenance_freq"
    ]
    
    # Handle Missing Values
    for c in df.select_dtypes(include=np.number).columns: 
        df[c].fillna(df[c].median(), inplace=True)
    for c in df.select_dtypes(include="object").columns:  
        df[c].fillna(df[c].mode()[0], inplace=True)
    df.drop_duplicates(subset="station_id", inplace=True)

    # ── 38 PRO-LEVEL ENGINEERED FEATURES ─────────────────────────────────────
    # Time & Age Metrics
    df["F01_age_yrs"] = 2024 - df["install_year"]
    df["F02_age_squared"] = df["F01_age_yrs"] ** 2
    df["F03_is_new_station"] = (df["F01_age_yrs"] <= 3).astype(int)
    
    # Efficiency & Financials
    df["F04_usage_per_capacity"] = df["usage_per_day"] / df["capacity_kw"].clip(1)
    df["F05_capacity_utilization"] = (df["F04_usage_per_capacity"] * 100).round(2)
    df["F06_cost_efficiency"] = (df["usage_per_day"] / df["cost_per_kwh"].clip(0.01)).round(2)
    df["F07_revenue_proxy"] = (df["usage_per_day"] * df["cost_per_kwh"] * 30).round(2)
    df["F08_revenue_per_spot"] = (df["F07_revenue_proxy"] / df["parking_spots"].clip(1)).round(2)
    df["F09_usage_per_spot"] = (df["usage_per_day"] / df["parking_spots"].clip(1)).round(2)
    
    # Weighted Master Score
    df["F10_performance_score"] = ((df["usage_per_day"]/100)*0.35 + (1-df["cost_per_kwh"]/0.5)*0.20 + (df["rating"]/5)*0.25).round(4)
    
    # Geospatial Transformations
    df["F11_dist_squared"] = df["dist_to_city_km"] ** 2
    df["F12_log_dist"] = np.log1p(df["dist_to_city_km"])
    df["F13_abs_lat"] = df["lat"].abs()
    
    # Categorical Encodings
    df["F14_is_247"] = (df["availability"] == "24/7").astype(int)
    df["F15_is_renewable"] = (df["renewable"] == "Yes").astype(int)
    df["F16_maintenance_score"] = df["maintenance_freq"].map({"Monthly": 3, "Quarterly": 2, "Annually": 1}).fillna(1)
    df["F17_maint_x_rating"] = df["F16_maintenance_score"] * df["rating"]
    
    # Hardware Capabilities
    df["F18_num_connectors"] = df["connector_types"].str.split(",").apply(len)
    df["F19_has_ccs"] = df["connector_types"].str.contains("CCS", case=False).astype(int)
    df["F20_has_tesla"] = df["connector_types"].str.contains("Tesla", case=False).astype(int)
    df["F21_charger_type_num"] = df["charger_type"].map({"AC Level 1": 1, "AC Level 2": 2, "DC Fast Charger": 3}).fillna(1)
    df["F22_is_fast_charger"] = (df["F21_charger_type_num"] == 3).astype(int)
    
    # Interaction Features
    df["F23_capacity_x_247"] = df["capacity_kw"] * df["F14_is_247"]
    df["F24_renewable_x_rating"] = df["F15_is_renewable"] * df["rating"]
    df["F25_age_x_usage"] = df["F01_age_yrs"] * df["usage_per_day"]
    df["F26_cost_x_dist"] = df["cost_per_kwh"] * df["dist_to_city_km"]
    df["F27_parking_x_capacity"] = df["parking_spots"] * df["capacity_kw"]
    
    # Normalization (Log Transforms)
    df["F28_log_usage"] = np.log1p(df["usage_per_day"])
    df["F29_log_capacity"] = np.log1p(df["capacity_kw"])
    df["F30_log_revenue"] = np.log1p(df["F07_revenue_proxy"])
    
    # Environmental Metrics
    df["F31_operator_encoded"] = LabelEncoder().fit_transform(df["operator"])
    df["F32_co2_saved_kg"] = (df["usage_per_day"] * 7.5 * 30).round(1)
    
    # F33-F35: Advanced K-Means Clustering
    clust_feats = ["usage_per_day", "cost_per_kwh", "capacity_kw", "rating", "dist_to_city_km"]
    X_sc = StandardScaler().fit_transform(df[clust_feats])
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["F33_cluster"] = km.fit_predict(X_sc).astype(str) # String format for Plotly discrete colors
    
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_sc)
    df["F34_pca1"], df["F35_pca2"] = pca_coords[:, 0], pca_coords[:, 1]

    # F36-F38: Advanced Anomaly Detection (Isolation Forest)
    iso = IsolationForest(contamination=0.05, random_state=42)
    df["F36_is_anomaly"] = (iso.fit_predict(X_sc) == -1).astype(int)
    df["F37_zscore"] = np.abs((df["usage_per_day"] - df["usage_per_day"].mean()) / df["usage_per_day"].std())
    df["F38_anomaly_type"] = np.where(df["F36_is_anomaly"] == 1, "Flagged Outlier", "Normal Data")

    return df

df = load_and_engineer_data()

# ─────────────────────────────────────────────────────────────────────────────
# 3. SIDEBAR CONTROLS
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2885/2885068.png", width=50)
st.sidebar.title("⚡ EV Control Panel")
st.sidebar.markdown("Filter the global dataset dynamically.")

selected_op = st.sidebar.multiselect("Network Operator", df["operator"].unique(), default=df["operator"].unique()[:4])
filtered_df = df[df["operator"].isin(selected_op)]

st.sidebar.divider()
st.sidebar.subheader("Export Center")
st.sidebar.markdown("Download the fully processed dataset with all 38 engineered features included.")
csv = df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="📥 Download Data (.csv)", 
    data=csv, 
    file_name="ev_analytics_pro_data.csv", 
    mime="text/csv"
)

# ─────────────────────────────────────────────────────────────────────────────
# 4. MAIN DASHBOARD: 3D GRAPHICS & ML
# ─────────────────────────────────────────────────────────────────────────────
st.title("Electric Vehicle Network Analytics")
st.markdown("Advanced data mining and 3D modeling powered by **38 engineered features**, **K-Means Clustering**, and **Anomaly Detection**.")

# 5 Sleek Tabs for the UI
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🚗 3D EV Showroom", 
    "🌍 3D Geospatial Map", 
    "🧊 Data Clustering", 
    "🚨 Anomaly Detection", 
    "🧠 Machine Learning Engine"
])

# ── TAB 1: 3D EV SHOWROOM (REALISTIC CAR MODEL) ──────────────────────────────
with tab1:
    st.subheader("Interactive WebGL Vehicle Rendering")
    st.markdown("Drag to rotate, scroll to zoom in/out. Fully rendered in real-time.")
    
    # Injecting HTML/JS to render the 3D model (Using a realistic Ferrari GLB from a reliable CDN)
    components.html('''
        <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.1.1/model-viewer.min.js"></script>
        <model-viewer 
            src="https://cdn.jsdelivr.net/gh/mrdoob/three.js@master/examples/models/gltf/ferrari.glb" 
            alt="A realistic 3D model of an Electric Vehicle" 
            auto-rotate 
            camera-controls 
            environment-image="neutral"
            shadow-intensity="2"
            exposure="1.2"
            camera-orbit="45deg 55deg 3m"
            style="width: 100%; height: 550px; background: radial-gradient(circle, #2b2b2b 0%, #0d1117 100%); border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.5);">
        </model-viewer>
    ''', height=600)

# ── TAB 2: 3D PYDECK MAP ─────────────────────────────────────────────────────
with tab2:
    st.subheader("Global 3D Station Density")
    st.markdown("Hold `Right-Click` to rotate and pitch the map. Hexagon height indicates station density.")
    
    layer = pdk.Layer(
        'HexagonLayer',
        data=filtered_df,
        get_position='[lon, lat]',
        radius=40000,          # Size of the hex bins
        elevation_scale=800,   # Scales the height for 3D visibility
        elevation_range=[0, 3000],
        pickable=True,
        extruded=True,
        get_fill_color="[230, 57, 70, 200]" # Match the Red UI aesthetic
    )
    
    view_state = pdk.ViewState(
        latitude=filtered_df["lat"].mean(),
        longitude=filtered_df["lon"].mean(),
        zoom=3.5,
        pitch=50,
        bearing=-15
    )
    
    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/dark-v10',
        layers=[layer], 
        initial_view_state=view_state, 
        tooltip={"text": "Station Density Elevation: {elevationValue}"}
    ))

# ── TAB 3: 3D PLOTLY GRAPHS ──────────────────────────────────────────────────
with tab3:
    st.subheader("Interactive 3D Feature Clustering (K-Means)")
    st.markdown("Explore how Cost, Capacity, and Daily Usage interact. Data is grouped into 4 distinct clusters.")
    
    fig_3d = px.scatter_3d(
        filtered_df, 
        x='cost_per_kwh', 
        y='capacity_kw', 
        z='usage_per_day', 
        color='F33_cluster',
        size='parking_spots',
        hover_name='operator',
        template='plotly_dark',
        color_discrete_sequence=PALETTE
    )
    fig_3d.update_layout(
        scene=dict(xaxis_title='Cost ($)', yaxis_title='Capacity (kW)', zaxis_title='Usage/Day'),
        margin=dict(l=0, r=0, b=0, t=0)
    )
    st.plotly_chart(fig_3d, use_container_width=True)

# ── TAB 4: ANOMALY DETECTION (ISOLATION FOREST) ──────────────────────────────
with tab4:
    st.subheader("Outlier Detection Analysis")
    st.markdown("Using Isolation Forest to automatically flag stations with abnormal operational patterns.")
    
    colA, colB = st.columns([1, 2])
    
    with colA:
        anomaly_count = filtered_df["F36_is_anomaly"].sum()
        normal_count = len(filtered_df) - anomaly_count
        st.metric(label="Total Anomalies Detected", value=anomaly_count)
        st.metric(label="Normal Stations", value=normal_count)
        st.success(f"Anomaly Rate: {(anomaly_count/len(filtered_df)*100):.1f}%")
        
    with colB:
        fig_anom = px.scatter(
            filtered_df, 
            x="cost_per_kwh", 
            y="usage_per_day", 
            color="F38_anomaly_type",
            size="capacity_kw",
            hover_name="station_id",
            template="plotly_dark",
            color_discrete_map={"Normal Data": "#2a9d8f", "Flagged Outlier": "#e63946"},
            title="Usage vs. Cost (Anomalies Highlighted)"
        )
        st.plotly_chart(fig_anom, use_container_width=True)

# ── TAB 5: PREDICTIVE MODEL & 38 FEATURES ────────────────────────────────────
with tab5:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Feature Engineering Pipeline")
        st.markdown(f"**{len([c for c in df.columns if c.startswith('F')])}** features generated.")
        eng_features = [c for c in df.columns if c.startswith("F")]
        st.dataframe(pd.DataFrame({"Engineered Column Name": eng_features}), height=450, use_container_width=True)
        
    with col2:
        st.subheader("Predictive Analytics (Random Forest)")
        st.markdown("Predicting `usage_per_day` using a subset of our newly engineered features.")
        
        # Supervised ML Model
        ml_features = ["F01_age_yrs", "F06_cost_efficiency", "F11_dist_squared", "F16_maintenance_score", "F22_is_fast_charger", "capacity_kw"]
        X = df[ml_features].dropna()
        y = df.loc[X.index, "usage_per_day"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
        
        # Scoring metrics ($R^2$, MAE, RMSE)
        c1, c2, c3 = st.columns(3)
        c1.metric("Model $R^2$ Score", f"{r2_score(y_test, preds):.3f}")
        c2.metric("Mean Abs Error (MAE)", f"{mean_absolute_error(y_test, preds):.2f}")
        c3.metric("Root Mean Sq Error", f"{np.sqrt(mean_squared_error(y_test, preds)):.2f}")
        
        # Feature Importance Chart
        fig_imp = px.bar(
            x=rf.feature_importances_, 
            y=ml_features, 
            orientation='h', 
            template='plotly_dark',
            title='Random Forest Feature Importance',
            labels={'x': 'Relative Importance', 'y': 'Input Feature'},
            color_discrete_sequence=["#e9c46a"]
        )
        fig_imp.update_layout(margin=dict(l=0, r=0, b=0, t=40))
        st.plotly_chart(fig_imp, use_container_width=True)
