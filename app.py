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
# 1. PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Pro EV Analytics", page_icon="⚡", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA PROCESSING & 38 ENGINEERED FEATURES
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_and_engineer_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "detailed_ev_charging_stations.csv")
    
    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset missing! Place 'detailed_ev_charging_stations.csv' in the same folder.")
        st.stop()

    df_raw = pd.read_csv(DATA_PATH)
    df = df_raw.copy()
    
    # Standardize column names (critical: 'lon' must be used for PyDeck compatibility)
    df.columns = [
        "station_id", "lat", "lon", "address", "charger_type", "cost_per_kwh", 
        "availability", "dist_to_city_km", "usage_per_day", "operator", "capacity_kw", 
        "connector_types", "install_year", "renewable", "rating", "parking_spots", "maintenance_freq"
    ]
    
    for c in df.select_dtypes(include=np.number).columns: 
        df[c].fillna(df[c].median(), inplace=True)
    for c in df.select_dtypes(include="object").columns:  
        df[c].fillna(df[c].mode()[0], inplace=True)
    df.drop_duplicates(subset="station_id", inplace=True)

    # ── 38 ENGINEERED FEATURES (PRO LEVEL) ───────────────────────────────────
    CURRENT_YEAR = 2024
    df["F01_age_yrs"] = CURRENT_YEAR - df["install_year"]
    df["F02_age_squared"] = df["F01_age_yrs"] ** 2
    df["F03_is_new"] = (df["F01_age_yrs"] <= 3).astype(int)
    
    df["F04_usage_per_cap"] = df["usage_per_day"] / df["capacity_kw"].clip(1)
    df["F05_cap_utilization"] = (df["F04_usage_per_cap"] * 100).round(2)
    df["F06_cost_efficiency"] = (df["usage_per_day"] / df["cost_per_kwh"].clip(0.01)).round(2)
    
    df["F07_revenue_proxy"] = (df["usage_per_day"] * df["cost_per_kwh"] * 30).round(2)
    df["F08_rev_per_spot"] = (df["F07_revenue_proxy"] / df["parking_spots"].clip(1)).round(2)
    df["F09_use_per_spot"] = (df["usage_per_day"] / df["parking_spots"].clip(1)).round(2)
    
    df["F10_perf_score"] = ((df["usage_per_day"]/100)*0.35 + (1-df["cost_per_kwh"]/0.5)*0.20 + (df["rating"]/5)*0.25).round(4)
    
    df["F11_dist_squared"] = df["dist_to_city_km"] ** 2
    df["F12_log_dist"] = np.log1p(df["dist_to_city_km"])
    df["F13_abs_lat"] = df["lat"].abs()
    
    df["F14_is_247"] = (df["availability"] == "24/7").astype(int)
    df["F15_is_renewable"] = (df["renewable"] == "Yes").astype(int)
    
    df["F16_maint_score"] = df["maintenance_freq"].map({"Monthly": 3, "Quarterly": 2, "Annually": 1}).fillna(1)
    df["F17_maint_x_rating"] = df["F16_maint_score"] * df["rating"]
    
    df["F18_num_connectors"] = df["connector_types"].str.split(",").apply(len)
    df["F19_has_ccs"] = df["connector_types"].str.contains("CCS", case=False).astype(int)
    df["F20_has_tesla"] = df["connector_types"].str.contains("Tesla", case=False).astype(int)
    
    df["F21_charger_type_num"] = df["charger_type"].map({"AC Level 1": 1, "AC Level 2": 2, "DC Fast Charger": 3}).fillna(1)
    df["F22_is_fast"] = (df["F21_charger_type_num"] == 3).astype(int)
    
    df["F23_cap_x_247"] = df["capacity_kw"] * df["F14_is_247"]
    df["F24_renew_x_rate"] = df["F15_is_renewable"] * df["rating"]
    df["F25_age_x_use"] = df["F01_age_yrs"] * df["usage_per_day"]
    df["F26_cost_x_dist"] = df["cost_per_kwh"] * df["dist_to_city_km"]
    df["F27_park_x_cap"] = df["parking_spots"] * df["capacity_kw"]
    
    df["F28_log_use"] = np.log1p(df["usage_per_day"])
    df["F29_log_cap"] = np.log1p(df["capacity_kw"])
    df["F30_log_rev"] = np.log1p(df["F07_revenue_proxy"])
    
    df["F31_op_encoded"] = LabelEncoder().fit_transform(df["operator"])
    df["F32_co2_saved_kg"] = (df["usage_per_day"] * 7.5 * 30).round(1)
    
    # F33-F35: Clustering Features
    clust_feats = ["usage_per_day", "cost_per_kwh", "capacity_kw", "rating", "dist_to_city_km"]
    X_sc = StandardScaler().fit_transform(df[clust_feats])
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["F33_cluster"] = km.fit_predict(X_sc).astype(str) # String for categorical coloring in Plotly
    
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_sc)
    df["F34_pca1"], df["F35_pca2"] = pca_coords[:, 0], pca_coords[:, 1]

    # F36-F38: Anomaly Detection Features
    iso = IsolationForest(contamination=0.05, random_state=42)
    df["F36_is_anomaly"] = (iso.fit_predict(X_sc) == -1).astype(int)
    df["F37_zscore"] = np.abs((df["usage_per_day"] - df["usage_per_day"].mean()) / df["usage_per_day"].std())
    df["F38_anomaly_type"] = np.where(df["F36_is_anomaly"] == 1, "Flagged Outlier", "Normal Data")

    return df

df = load_and_engineer_data()

# ─────────────────────────────────────────────────────────────────────────────
# 3. SIDEBAR CONTROLS & DATA EXPORT
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("⚡ Pro Settings")
st.sidebar.markdown("Filter the dataset dynamically.")
selected_op = st.sidebar.multiselect("Operator", df["operator"].unique(), default=df["operator"].unique()[:4])
filtered_df = df[df["operator"].isin(selected_op)]

st.sidebar.divider()
st.sidebar.subheader("Export Center")
csv = df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="📥 Download 38-Feature Dataset", 
    data=csv, 
    file_name="ev_pro_data.csv", 
    mime="text/csv"
)

# ─────────────────────────────────────────────────────────────────────────────
# 4. MAIN DASHBOARD: 3D GRAPHICS & ML
# ─────────────────────────────────────────────────────────────────────────────
st.title("Enterprise EV Analytics & 3D Modeling")
st.markdown("Advanced analytics powered by 38 engineered features, Machine Learning, and interactive WebGL graphics.")

tab1, tab2, tab3, tab4 = st.tabs(["🌍 3D Geospatial Map", "🧊 3D Cluster Analysis", "🚗 3D EV Showroom", "🤖 ML & Features"])

# ── TAB 1: 3D PYDECK MAP ─────────────────────────────────────────────────────
with tab1:
    st.subheader("Global 3D Station Density")
    st.markdown("Hold `Right-Click` to rotate and pitch the map. Hexagon height indicates station capacity/density.")
    
    # 3D Hexagon Layer using PyDeck
    layer = pdk.Layer(
        'HexagonLayer',
        data=filtered_df,
        get_position='[lon, lat]',
        radius=50000,          # Size of the hex bins
        elevation_scale=1000,  # Scales the height for visibility
        elevation_range=[0, 3000],
        pickable=True,
        extruded=True,
        get_fill_color="[255, 255 - (elevationValue * 10), 0, 200]" # Gradient color
    )
    
    view_state = pdk.ViewState(
        latitude=filtered_df["lat"].mean(),
        longitude=filtered_df["lon"].mean(),
        zoom=3,
        pitch=45,
        bearing=10
    )
    
    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/dark-v10',
        layers=[layer], 
        initial_view_state=view_state, 
        tooltip={"text": "Elevation Density: {elevationValue}"}
    ))

# ── TAB 2: 3D PLOTLY GRAPHS ──────────────────────────────────────────────────
with tab2:
    st.subheader("3D Interactive Feature Clustering")
    st.markdown("Explore how Usage, Cost, and Capacity interact in three dimensions. Rotate with your mouse.")
    
    fig_3d = px.scatter_3d(
        filtered_df, 
        x='cost_per_kwh', 
        y='capacity_kw', 
        z='usage_per_day', 
        color='F33_cluster',
        size='parking_spots',
        hover_name='operator',
        template='plotly_dark',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_3d.update_layout(
        scene=dict(xaxis_title='Cost ($)', yaxis_title='Capacity (kW)', zaxis_title='Usage/Day'),
        margin=dict(l=0, r=0, b=0, t=0)
    )
    st.plotly_chart(fig_3d, use_container_width=True)

# ── TAB 3: 3D EV SHOWROOM ────────────────────────────────────────────────────
with tab3:
    st.subheader("Interactive 3D WebGL Vehicle")
    st.markdown("Drag to rotate, scroll to zoom in/out. Powered by `<model-viewer>`.")
    
    # Injecting HTML/JS to render the 3D model
    components.html('''
        <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.1.1/model-viewer.min.js"></script>
        <model-viewer 
            src="https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/Buggy/glTF-Binary/Buggy.glb" 
            alt="A 3D model of a vehicle" 
            auto-rotate 
            camera-controls 
            environment-image="neutral"
            shadow-intensity="1"
            style="width: 100%; height: 500px; background-color: #0d1117; border-radius: 10px;">
        </model-viewer>
    ''', height=550)

# ── TAB 4: PREDICTIVE MODEL & 38 FEATURES ────────────────────────────────────
with tab4:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Feature Engineering Pipeline")
        st.markdown("The complete list of 38 features generated automatically.")
        # Isolate the engineered features into a clean table for display
        eng_features = [c for c in df.columns if c.startswith("F")]
        st.dataframe(pd.DataFrame({"Feature Name": eng_features}), height=450, use_container_width=True)
        
    with col2:
        st.subheader("Random Forest Predictor")
        st.markdown("Predicting `usage_per_day` using our newly engineered feature set.")
        
        # Select highest-impact features for the ML model
        ml_features = ["F01_age_yrs", "F06_cost_efficiency", "F11_dist_squared", "F16_maint_score", "F22_is_fast", "capacity_kw"]
        X = df[ml_features].dropna()
        y = df.loc[X.index, "usage_per_day"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Model R² Score", f"{r2_score(y_test, preds):.3f}")
        c2.metric("Mean Absolute Error", f"{mean_absolute_error(y_test, preds):.2f}")
        c3.metric("Root Mean Squared Error", f"{np.sqrt(mean_squared_error(y_test, preds)):.2f}")
        
        # Plotly Feature Importance Bar Chart
        fig = px.bar(
            x=rf.feature_importances_, 
            y=ml_features, 
            orientation='h', 
            template='plotly_dark',
            title='Feature Importance Engine',
            labels={'x': 'Relative Importance', 'y': 'Input Feature'}
        )
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
        st.plotly_chart(fig, use_container_width=True)
