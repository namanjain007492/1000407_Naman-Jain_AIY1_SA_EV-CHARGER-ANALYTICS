import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from mlxtend.frequent_patterns import apriori, association_rules
import io

# ----------------------------------------
# PAGE CONFIGURATION & THEME
# ----------------------------------------
st.set_page_config(
    page_title="EV Smart Analytics & Interactive Car Experience",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark futuristic theme
st.markdown("""
    <style>
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    h1, h2, h3 { color: #58a6ff; }
    .stButton>button {
        background-color: #238636;
        color: white;
        border-radius: 8px;
        border: 1px solid #2ea043;
    }
    .stButton>button:hover { background-color: #2ea043; }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------
# ⚙️ STEP 1-3: DATA LOADING & PREPROCESSING
# ----------------------------------------
@st.cache_data
def load_and_preprocess_data():
    # 1. Load Data
    try:
        df = pd.read_csv("detailed_ev_charging_stations.csv")
    except FileNotFoundError:
        st.error("Error: 'detailed_ev_charging_stations.csv' not found. Please ensure it's in the same directory.")
        return None

    # 2. Data Cleaning
    df = df.drop_duplicates(subset=['Station ID'])
    
    # Fill missing values
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    cat_cols = df.select_dtypes(include=['object']).columns
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

    # 3. Feature Engineering
    df['Cost Efficiency'] = df['Usage Stats (avg users/day)'] / df['Cost (USD/kWh)'].replace(0, 0.01) # Avoid division by zero
    
    # Demand Level
    def demand_level(x):
        if x > 75: return 'High'
        elif x >= 40: return 'Medium'
        else: return 'Low'
    df['Demand Level'] = df['Usage Stats (avg users/day)'].apply(demand_level)

    # Distance Category
    def distance_cat(x):
        if x < 10: return 'Near'
        elif x <= 30: return 'Mid'
        else: return 'Far'
    df['Distance Category'] = df['Distance to City (km)'].apply(distance_cat)

    # Encoding
    le = LabelEncoder()
    df['Charger_Type_Encoded'] = le.fit_transform(df['Charger Type'])
    df['Operator_Encoded'] = le.fit_transform(df['Station Operator'])
    df['Renewable_Encoded'] = df['Renewable Energy Source'].apply(lambda x: 1 if x == 'Yes' else 0)

    return df

df = load_and_preprocess_data()

if df is not None:
    # ----------------------------------------
    # 🧭 STEP 14: APP NAVIGATION STRUCTURE
    # ----------------------------------------
    st.sidebar.title("⚡ EV Dashboard")
    st.sidebar.markdown("---")
    
    menu = st.sidebar.radio("Navigate", [
        "📊 Dashboard & EDA", 
        "🗺️ Map View", 
        "🤖 Clustering & ML", 
        "🚨 Anomaly Detection", 
        "🔗 Association Rules",
        "📈 Demand Prediction",
        "🚗 3D EV Experience",
        "📘 Learn EV",
        "🎮 Fun Zone"
    ])

    st.sidebar.markdown("---")
    st.sidebar.info("Student Sample Project: EV Analytics")

    # Filters (applied globally to Dashboard and Map)
    st.sidebar.subheader("🎛️ Global Filters")
    cost_filter = st.sidebar.slider("Max Cost (USD/kWh)", float(df['Cost (USD/kWh)'].min()), float(df['Cost (USD/kWh)'].max()), float(df['Cost (USD/kWh)'].max()))
    charger_filter = st.sidebar.selectbox("Charger Type", ['All'] + list(df['Charger Type'].unique()))
    operator_filter = st.sidebar.selectbox("Operator", ['All'] + list(df['Station Operator'].unique()))

    # Apply filters
    filtered_df = df[df['Cost (USD/kWh)'] <= cost_filter]
    if charger_filter != 'All': filtered_df = filtered_df[filtered_df['Charger Type'] == charger_filter]
    if operator_filter != 'All': filtered_df = filtered_df[filtered_df['Station Operator'] == operator_filter]

    # ----------------------------------------
    # MENU: DASHBOARD & EDA
    # ----------------------------------------
    if menu == "📊 Dashboard & EDA":
        st.title("📊 EV Analytics Dashboard")
        
        # Top Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Stations", len(filtered_df))
        col2.metric("Avg Usage/Day", f"{filtered_df['Usage Stats (avg users/day)'].mean():.1f}")
        col3.metric("Avg Cost ($/kWh)", f"{filtered_df['Cost (USD/kWh)'].mean():.2f}")
        col4.metric("Avg Rating", f"{filtered_df['Reviews (Rating)'].mean():.1f} / 5.0")

        st.markdown("### 📊 Exploratory Data Analysis")
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data Preview", "Usage Dist", "Cost vs Operator", "Capacity vs Usage", "Heatmap"])
        
        with tab1:
            st.dataframe(filtered_df.head(10))
            # Export Feature
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Filtered Dataset", data=csv, file_name='filtered_ev_data.csv', mime='text/csv')

        with tab2:
            fig = px.histogram(filtered_df, x="Usage Stats (avg users/day)", color="Demand Level", title="Usage Distribution", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            
        with tab3:
            fig = px.box(filtered_df, x="Station Operator", y="Cost (USD/kWh)", color="Station Operator", title="Cost by Operator", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

        with tab4:
            fig = px.scatter(filtered_df, x="Charging Capacity (kW)", y="Usage Stats (avg users/day)", color="Charger Type", size="Reviews (Rating)", hover_data=["Station ID"], template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

        with tab5:
            corr = filtered_df[['Cost (USD/kWh)', 'Distance to City (km)', 'Usage Stats (avg users/day)', 'Charging Capacity (kW)', 'Reviews (Rating)']].corr()
            fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap", template="plotly_dark", color_continuous_scale="Viridis")
            st.plotly_chart(fig, use_container_width=True)

        # AI Recommendation
        st.markdown("### 🤖 AI Station Recommendation")
        recommendation = filtered_df.sort_values(by=['Reviews (Rating)', 'Cost (USD/kWh)', 'Usage Stats (avg users/day)'], ascending=[False, True, False]).head(1)
        if not recommendation.empty:
            st.success(f"**Recommended Station:** {recommendation['Station ID'].values[0]} | **Operator:** {recommendation['Station Operator'].values[0]} | **Rating:** {recommendation['Reviews (Rating)'].values[0]} ⭐ | **Cost:** ${recommendation['Cost (USD/kWh)'].values[0]}/kWh")

    # ----------------------------------------
    # MENU: MAP VIEW
    # ----------------------------------------
    elif menu == "🗺️ Map View":
        st.title("🗺️ Station Locations")
        fig = px.scatter_mapbox(filtered_df, lat="Latitude", lon="Longitude", color="Demand Level",
                                size="Usage Stats (avg users/day)", hover_name="Station ID",
                                hover_data=["Cost (USD/kWh)", "Reviews (Rating)", "Station Operator"],
                                color_discrete_map={"High": "red", "Medium": "yellow", "Low": "green"},
                                zoom=2, height=600)
        fig.update_layout(mapbox_style="carto-darkmatter", margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)

    # ----------------------------------------
    # MENU: CLUSTERING
    # ----------------------------------------
    elif menu == "🤖 Clustering & ML":
        st.title("🤖 Station Clustering (K-Means)")
        features = ['Usage Stats (avg users/day)', 'Cost (USD/kWh)', 'Charging Capacity (kW)']
        X = df[features].dropna()
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        k = st.slider("Select Number of Clusters (K)", 2, 10, 3)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        plot_df = X.copy()
        plot_df['Cluster'] = clusters.astype(str)

        fig = px.scatter_3d(plot_df, x='Usage Stats (avg users/day)', y='Cost (USD/kWh)', z='Charging Capacity (kW)',
                            color='Cluster', title=f"K-Means Clustering (K={k})", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    # ----------------------------------------
    # MENU: ANOMALY DETECTION
    # ----------------------------------------
    elif menu == "🚨 Anomaly Detection":
        st.title("🚨 Usage Anomaly Detection (Z-Score)")
        
        df_anomaly = df.copy()
        mean_usage = df_anomaly['Usage Stats (avg users/day)'].mean()
        std_usage = df_anomaly['Usage Stats (avg users/day)'].std()
        
        df_anomaly['Z-Score'] = (df_anomaly['Usage Stats (avg users/day)'] - mean_usage) / std_usage
        threshold = st.slider("Z-Score Threshold", 1.0, 5.0, 2.5)
        
        df_anomaly['Anomaly'] = df_anomaly['Z-Score'].apply(lambda x: 'Anomaly (High/Low)' if abs(x) > threshold else 'Normal')
        
        fig = px.scatter(df_anomaly, x="Station ID", y="Usage Stats (avg users/day)", color="Anomaly",
                         color_discrete_map={'Normal': '#58a6ff', 'Anomaly (High/Low)': '#ff4444'},
                         title="Station Usage Anomalies", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("### Anomaly Data")
        st.dataframe(df_anomaly[df_anomaly['Anomaly'] != 'Normal'][['Station ID', 'Usage Stats (avg users/day)', 'Z-Score']])

    # ----------------------------------------
    # MENU: ASSOCIATION RULES
    # ----------------------------------------
    elif menu == "🔗 Association Rules":
        st.title("🔗 Association Rule Mining (Apriori)")
        st.markdown("Discover hidden patterns like: *Low Cost + Fast Charger -> High Demand*")
        
        # Prepare transactional data
        assoc_df = df[['Demand Level', 'Distance Category', 'Charger Type']].copy()
        assoc_df['Cost Level'] = pd.qcut(df['Cost (USD/kWh)'], q=3, labels=['Low Cost', 'Mid Cost', 'High Cost'])
        
        # One-hot encode
        basket = pd.get_dummies(assoc_df).astype(bool)
        
        min_support = st.slider("Minimum Support", 0.05, 0.5, 0.1)
        frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)
        
        if not frequent_itemsets.empty:
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
            rules = rules.sort_values('lift', ascending=False).head(20)
            
            # Formatting for display
            rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
            rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
            
            st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
        else:
            st.warning("No rules found. Try lowering the support threshold.")

    # ----------------------------------------
    # MENU: DEMAND PREDICTION
    # ----------------------------------------
    elif menu == "📈 Demand Prediction":
        st.title("📈 Demand Prediction (Machine Learning)")
        
        features = ['Cost (USD/kWh)', 'Charging Capacity (kW)', 'Distance to City (km)']
        target = 'Usage Stats (avg users/day)'
        
        X = df[features]
        y = df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        st.subheader("Predict Usage for a New Station")
        col1, col2, col3 = st.columns(3)
        p_cost = col1.number_input("Cost ($/kWh)", 0.1, 1.0, 0.3)
        p_cap = col2.number_input("Capacity (kW)", 10, 350, 50)
        p_dist = col3.number_input("Distance to City (km)", 0.0, 100.0, 10.0)
        
        if st.button("Predict Expected Users/Day"):
            pred = model.predict([[p_cost, p_cap, p_dist]])[0]
            st.success(f"Predicted Average Users per Day: **{pred:.2f}**")
            
        fig = px.scatter(x=y_test, y=predictions, labels={'x': 'Actual Usage', 'y': 'Predicted Usage'}, 
                         title="Model Performance: Actual vs Predicted", template="plotly_dark")
        fig.add_shape(type="line", x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max(), line=dict(color="red", dash="dash"))
        st.plotly_chart(fig, use_container_width=True)

    # ----------------------------------------
    # MENU: 3D EV CAR EXPERIENCE
    # ----------------------------------------
    elif menu == "🚗 3D EV Experience":
        st.title("🚗 Interactive 3D EV Experience")
        st.markdown("Rotate, zoom, and explore the EV components.")
        
        # Embedding Google's Model Viewer with an open-source car model
        html_code = """
        <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.1.1/model-viewer.min.js"></script>
        <div style="display: flex; justify-content: center; background-color: #161b22; border-radius: 10px; padding: 20px;">
            <model-viewer 
                src="https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/ToyCar/glTF-Binary/ToyCar.glb" 
                alt="A 3D model of a car" 
                auto-rotate 
                camera-controls 
                style="width: 100%; height: 500px; outline: none;">
                
                <button slot="hotspot-battery" data-position="0 0.1 0" data-normal="0 1 0" style="background: #58a6ff; color: white; border: none; border-radius: 5px; padding: 5px; font-weight: bold;">Battery Pack</button>
                <button slot="hotspot-motor" data-position="0 0.3 0.8" data-normal="0 1 0" style="background: #238636; color: white; border: none; border-radius: 5px; padding: 5px; font-weight: bold;">Electric Motor</button>
            </model-viewer>
        </div>
        """
        st.components.v1.html(html_code, height=600)
        
        st.info("**Key Components:**\n- **Battery Pack:** Stores electrical energy.\n- **Electric Motor:** Drives the wheels.\n- **Inverter:** Converts DC to AC power.\n- **Cooling System:** Regulates thermal temps.")

    # ----------------------------------------
    # MENU: LEARN EV
    # ----------------------------------------
    elif menu == "📘 Learn EV":
        st.title("📘 Learn About Electric Vehicles")
        
        with st.expander("🤔 How does an EV work?"):
            st.write("Unlike traditional vehicles that use internal combustion engines, electric vehicles run on electric motors powered by rechargeable battery packs. They have fewer moving parts, resulting in lower maintenance and zero tailpipe emissions.")
            
        with st.expander("⚡ Types of EV Chargers"):
            st.markdown("""
            - **Level 1 (AC):** Uses standard 120V outlet. Slowest charging (~3-5 miles range per hour).
            - **Level 2 (AC):** Uses 240V. Common in homes and public stations (~15-30 miles range per hour).
            - **DC Fast Charging:** Direct current, bypasses onboard charger. Can charge up to 80% in 20-30 minutes.
            """)
            
        with st.expander("🌱 EV vs Petrol Comparison"):
            st.write("**Efficiency:** EVs convert ~77% of electrical energy to the wheels. Gasoline vehicles only convert ~12%-30% of energy stored in fuel.")

        st.subheader("💡 Fun EV Facts")
        st.success("- The first practical electric car was built in the 1880s!\n- Regenerative braking captures kinetic energy and stores it back in the battery.\n- EVs are completely silent, so manufacturers add artificial sounds at low speeds for pedestrian safety.")

    # ----------------------------------------
    # MENU: FUN ZONE (GAMES)
    # ----------------------------------------
    elif menu == "🎮 Fun Zone":
        st.title("🎮 EV Mini-Games & Quiz")
        
        st.subheader("Quiz: Test your EV Knowledge!")
        
        # Simple form for quiz to prevent auto-reloading weirdness
        with st.form("quiz_form"):
            q1 = st.radio("1. What does DC stand for in DC Fast Charging?", ("Direct Current", "Dual Charge", "Dynamic Current"), index=None)
            q2 = st.radio("2. Which component converts DC battery power to AC for the motor?", ("Alternator", "Inverter", "Radiator"), index=None)
            q3 = st.radio("3. Average efficiency of an EV motor is around:", ("30%", "50%", "90%"), index=None)
            
            submitted = st.form_submit_button("Submit Quiz")
            
            if submitted:
                score = 0
                if q1 == "Direct Current": score += 1
                if q2 == "Inverter": score += 1
                if q3 == "90%": score += 1
                
                if score == 3:
                    st.balloons()
                    
                st.write(f"### Your Score: {score}/3")
                if score == 3: 
                    st.success("🎉 You are an EV Expert!")
                else: 
                    st.info("Keep learning in the 'Learn EV' tab!")
