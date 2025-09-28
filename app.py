import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

# ---------------------------
# Load Baseline XGBoost Model
# ---------------------------
model = joblib.load('xgboost_pubg_baseline.pkl')

# ---------------------------
# Streamlit App Interface
# ---------------------------
st.title("PUBG Win Probability Predictor")
st.write("Predict a player's win probability (`winPlacePerc`) based on match stats.")

st.sidebar.header("Player Match Stats Input")

# Sample player types
player_type = st.sidebar.selectbox("Select Player Type", ("Aggressive", "Balanced", "Passive"))

# Set default stats based on player type
if player_type == "Aggressive":
    default_stats = {"kills":10, "damageDealt":1200, "walkDistance":2000, "rideDistance":800, "swimDistance":100, "heals":5, "boosts":3, "headshotKills":5, "killPlace":10, "numGroups":50, "maxPlace":50, "weaponsAcquired":10, "longestKill":250, "DBNOs":3}
elif player_type == "Balanced":
    default_stats = {"kills":5, "damageDealt":600, "walkDistance":1500, "rideDistance":500, "swimDistance":50, "heals":4, "boosts":2, "headshotKills":2, "killPlace":25, "numGroups":50, "maxPlace":50, "weaponsAcquired":8, "longestKill":150, "DBNOs":2}
else:  # Passive
    default_stats = {"kills":1, "damageDealt":200, "walkDistance":1000, "rideDistance":300, "swimDistance":20, "heals":2, "boosts":1, "headshotKills":0, "killPlace":50, "numGroups":50, "maxPlace":50, "weaponsAcquired":5, "longestKill":50, "DBNOs":0}

# Input sliders
kills = st.sidebar.slider("Kills", 0, 30, default_stats["kills"])
damageDealt = st.sidebar.slider("Damage Dealt", 0, 3000, default_stats["damageDealt"])
walkDistance = st.sidebar.slider("Walk Distance", 0.0, 5000.0, default_stats["walkDistance"])
rideDistance = st.sidebar.slider("Ride Distance", 0.0, 5000.0, default_stats["rideDistance"])
swimDistance = st.sidebar.slider("Swim Distance", 0.0, 1000.0, default_stats["swimDistance"])
heals = st.sidebar.slider("Heals", 0, 20, default_stats["heals"])
boosts = st.sidebar.slider("Boosts", 0, 10, default_stats["boosts"])
headshotKills = st.sidebar.slider("Headshot Kills", 0, 20, default_stats["headshotKills"])
killPlace = st.sidebar.slider("Kill Place", 1, 100, default_stats["killPlace"])
numGroups = st.sidebar.slider("Number of Groups", 1, 100, default_stats["numGroups"])
maxPlace = st.sidebar.slider("Max Place in Match", 1, 100, default_stats["maxPlace"])
weaponsAcquired = st.sidebar.slider("Weapons Acquired", 0, 50, default_stats["weaponsAcquired"])
longestKill = st.sidebar.slider("Longest Kill", 0.0, 1000.0, default_stats["longestKill"])
DBNOs = st.sidebar.slider("DBNOs", 0, 20, default_stats["DBNOs"])

# Feature engineering
totalDistance = walkDistance + rideDistance + swimDistance
killRatio = kills / totalDistance if totalDistance != 0 else 0
killsPerDistance = kills / totalDistance if totalDistance != 0 else 0
headshotRate = headshotKills / kills if kills != 0 else 0

# Prepare input DataFrame
input_df = pd.DataFrame({
    "killPlace": [killPlace],
    "numGroups": [numGroups],
    "maxPlace": [maxPlace],
    "walkDistance": [walkDistance],
    "totalDistance": [totalDistance],
    "killsPerDistance": [killsPerDistance],
    "damageDealt": [damageDealt],
    "kills": [kills],
    "weaponsAcquired": [weaponsAcquired],
    "longestKill": [longestKill],
    "killRatio": [killRatio],
    "boosts": [boosts],
    "rideDistance": [rideDistance],
    "DBNOs": [DBNOs]
})

# Prediction
if st.button("Predict Win Probability"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Win Probability (winPlacePerc): {prediction:.3f}")

# ---------------------------
# Feature Importance Plot
# ---------------------------
st.markdown("---")
st.subheader("Feature Importance (XGBoost)")

# Extract feature importance
importance = model.feature_importances_
features = input_df.columns
feat_imp_df = pd.DataFrame({"Feature": features, "Importance": importance}).sort_values(by="Importance", ascending=False)

# Plot
fig, ax = plt.subplots()
ax.barh(feat_imp_df["Feature"], feat_imp_df["Importance"])
ax.invert_yaxis()
st.pyplot(fig)

st.write("**Instructions:** Adjust the sliders in the sidebar to input player stats, select player type for defaults, then click 'Predict Win Probability'.")
