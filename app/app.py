import streamlit as st
import pandas as pd
import numpy as np
import xgboost
import joblib
import os
import plotly.graph_objects as go


st.set_page_config(
    page_title="PUBG | Winner Prediction",
    page_icon="🎮",
    layout="wide"
)


st.markdown("""
<style>

/* Background animated grid */
.stApp {
    background: radial-gradient(circle at top, #050505, #0a0f1c, #000);
    color: white;
}

/* Neon HUD Title */
.title {
    font-size: 55px;
    text-align: center;
    font-weight: 900;
    color: #00fff2;
    text-shadow: 0 0 10px #00fff2, 0 0 25px #0088ff;
    animation: flicker 2s infinite;
}

@keyframes flicker {
    0% {opacity: 1;}
    50% {opacity: 0.7;}
    100% {opacity: 1;}
}

/* Glass HUD panel */
.hud {
    background: rgba(0, 255, 255, 0.05);
    border: 1px solid rgba(0, 255, 255, 0.2);
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 0 20px rgba(0,255,255,0.2);
}

/* Neon button */
.stButton>button {
    background: linear-gradient(90deg, #00fff2, #0066ff);
    color: black;
    font-weight: bold;
    border-radius: 10px;
    box-shadow: 0 0 15px #00fff2;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #050a14;
}
 /* Input field labels (e.g. "Kills", "Match Duration", "Match Type") */
.stNumberInput label, .stSelectbox label {
    color: #00fff2 !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    text-shadow: 0 0 6px rgba(0, 255, 242, 0.5);
}           

</style>
""", unsafe_allow_html=True)

# =============================
# 📦 LOAD MODEL
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "pubg_best_model.pkl")

model = joblib.load(MODEL_PATH)

# =============================
# 🎮 TITLE
# =============================
st.markdown('<div class="title">🎮 Game Winner Prediction</div>', unsafe_allow_html=True)

st.write("🔵 Real-Time Battle Performance Analyzer + Win Prediction Engine")

# =============================
# 🎯 INPUT PANEL (GAME HUD STYLE)
# =============================
st.markdown("## 🧠 PLAYER STATS INPUT")

col1, col2, col3 = st.columns(3)

with col1:
    assists = st.number_input("Assists", 0)
    boosts = st.number_input("Boosts", 0)
    damageDealt = st.number_input("Damage Dealt", 0.0)
    DBNOs = st.number_input("DBNOs", 0)
    headshotKills = st.number_input("Headshot Kills", 0)
    heals = st.number_input("Heals", 0)
    killPlace = st.number_input("Kill Place", 1)
    killPoints = st.number_input("Kill Points", 0)

with col2:
    kills = st.number_input("Kills", 0)
    killStreaks = st.number_input("Kill Streaks", 0)
    longestKill = st.number_input("Longest Kill", 0.0)
    matchDuration = st.number_input("Match Duration", 0)
    matchType = st.selectbox(
        "Match Type",
        [
            "crashfpp", "crashtpp", "duo", "duo-fpp", "flarefpp", "flaretpp",
            "normal-duo", "normal-duo-fpp", "normal-solo", "normal-solo-fpp",
            "normal-squad", "normal-squad-fpp", "solo", "solo-fpp", "squad", "squad-fpp"
        ],
        index=15  # default: squad-fpp (most common match type in training data)
    )
    maxPlace = st.number_input("Max Place", 1)
    numGroups = st.number_input("Number of Groups", 1)
    rankPoints = st.number_input("Rank Points", 0)
    revives = st.number_input("Revives", 0)

with col3:
    rideDistance = st.number_input("Ride Distance", 0.0)
    roadKills = st.number_input("Road Kills", 0)
    swimDistance = st.number_input("Swim Distance", 0.0)
    teamKills = st.number_input("Team Kills", 0)
    vehicleDestroys = st.number_input("Vehicle Destroys", 0)
    walkDistance = st.number_input("Walk Distance", 0.0)
    weaponsAcquired = st.number_input("Weapons Acquired", 0)
    winPoints = st.number_input("Win Points", 0)

# =============================
# 🔤 ENCODE matchType
# =============================
# Must match the LabelEncoder mapping used during training (alphabetical order
# of the 16 categories present in the training dataset).
MATCH_TYPE_MAP = {
    "crashfpp": 0,
    "crashtpp": 1,
    "duo": 2,
    "duo-fpp": 3,
    "flarefpp": 4,
    "flaretpp": 5,
    "normal-duo": 6,
    "normal-duo-fpp": 7,
    "normal-solo": 8,
    "normal-solo-fpp": 9,
    "normal-squad": 10,
    "normal-squad-fpp": 11,
    "solo": 12,
    "solo-fpp": 13,
    "squad": 14,
    "squad-fpp": 15,
}

matchType_encoded = MATCH_TYPE_MAP[matchType]

# =============================
# 🔥 FEATURE VECTOR
# =============================
input_data = pd.DataFrame([[
    assists, boosts, damageDealt, DBNOs, headshotKills, heals,
    killPlace, killPoints, kills, killStreaks, longestKill,
    matchDuration, matchType_encoded, maxPlace, numGroups, rankPoints, revives,
    rideDistance, roadKills, swimDistance, teamKills,
    vehicleDestroys, walkDistance, weaponsAcquired, winPoints
]], columns=[
    "assists", "boosts", "damageDealt", "DBNOs", "headshotKills", "heals",
    "killPlace", "killPoints", "kills", "killStreaks", "longestKill",
    "matchDuration", "matchType", "maxPlace", "numGroups", "rankPoints", "revives",
    "rideDistance", "roadKills", "swimDistance", "teamKills",
    "vehicleDestroys", "walkDistance", "weaponsAcquired", "winPoints"
])

# =============================
# 🎯 RADAR CHART (AAA VISUAL)
# =============================
def radar_chart(data):
    labels = [
        "Assists", "Boosts", "Damage", "Kills", "Headshots",
        "Distance", "Survival", "Movement", "Utility"
    ]

    values = [
        assists,
        boosts,
        damageDealt / 100,
        kills,
        headshotKills,
        walkDistance / 1000,
        matchDuration / 1000,
        rideDistance / 1000,
        weaponsAcquired
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=labels,
        fill='toself',
        name='Player Profile',
        line=dict(color='#00fff2')
    ))

    fig.update_layout(
        polar=dict(
            bgcolor="#0a0f1c"
        ),
        paper_bgcolor="#000",
        font=dict(color="#00fff2")
    )

    return fig

# =============================
# 🚀 PREDICT BUTTON
# =============================
if st.button("⚡RUN PREDICTION ANALYSIS"):

    pred = model.predict(input_data)[0]

    win_percent = max(0, min(100, pred * 100))

    # =============================
    # 🏆 RANK SYSTEM
    # =============================
    if win_percent < 20:
        rank = "🥉 BRONZE"
    elif win_percent < 40:
        rank = "🥈 SILVER"
    elif win_percent < 60:
        rank = "🥇 GOLD"
    elif win_percent < 80:
        rank = "💎 DIAMOND"
    else:
        rank = "👑 CONQUEROR"

    # =============================
    # 🎯 RESULT HUD
    # =============================
    st.markdown("## 🔵 COMBAT ANALYSIS OUTPUT")

    colA, colB = st.columns(2)

    with colA:
        st.markdown('<div class="hud">', unsafe_allow_html=True)
        st.metric("WIN PROBABILITY", f"{win_percent:.2f}%")
        st.metric("PLAYER RANK", rank)
        st.progress(int(win_percent))
        st.markdown("</div>", unsafe_allow_html=True)

    with colB:
        st.plotly_chart(radar_chart(input_data))

    # =============================
    # 🧠 AI INSIGHT
    # =============================
    st.markdown("## 🧠 AI COMBAT INSIGHT")

    if win_percent > 70:
        st.success("🔥 ELITE PLAYER DETECTED — HIGH WINNING POTENTIAL")
    elif win_percent > 40:
        st.warning("⚔️ AVERAGE PERFORMANCE — IMPROVE POSITIONING & KILLS")
    else:
        st.error("📉 LOW SURVIVAL RATE — STRATEGY REQUIRED")

# =============================
# FOOTER
# =============================
st.markdown("---")
st.markdown("🎮 PUBG AI Winner | XGBoost-Powered Prediction Engine")

