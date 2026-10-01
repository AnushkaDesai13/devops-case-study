import streamlit as st
import joblib
from agent import DeliveryAgent

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="AI Food Delivery System",
    page_icon="🍔",
    layout="centered"
)

st.title("AI Food Delivery System")
st.subheader("Agentic AI Delivery Predictor")

# -------------------------------------------------
# Load saved model + encoders
# -------------------------------------------------
@st.cache_resource
def load_assets():
    model = joblib.load("delivery_model.pkl")
    encoders = joblib.load("label_encoders.pkl")
    return model, encoders

model, encoders = load_assets()

# -------------------------------------------------
# Inputs
# -------------------------------------------------
distance = st.slider("Distance (km)", 1.0, 15.0, 5.0, 0.1)
traffic = st.selectbox("Traffic Level", ["Low", "Medium", "High"])
weather = st.selectbox("Weather", ["Clear", "Rainy", "Stormy"])
premium = st.selectbox("Premium Order?", ["No", "Yes"])

# -------------------------------------------------
# Predict button
# -------------------------------------------------
if st.button("Predict & Assign"):

    order = {
        "distance_km": distance,
        "traffic_level": traffic,
        "weather": weather,
        "prep_time": 20,
        "partner_rating": 4.5,
        "partner_available": 1,
        "premium_order": 1 if premium == "Yes" else 0
    }

    # Your agent.py requires observe_order() then act()
    agent = DeliveryAgent(model, encoders)
    agent.observe_order(order)
    result = agent.act()

    st.success("Decision Complete")

    st.write("### Result")
    st.write("Assigned Partner:", result["partner"])
    st.write("Predicted ETA:", result["eta_min"], "minutes")
    st.write("Action Taken:", result["action"])

    st.write("### Agent Reasoning")
    for log in result["logs"]:
        st.write(log)