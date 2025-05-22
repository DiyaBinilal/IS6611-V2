# probabilistic output
import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load(r"C:\Users\Sajin Siyad\dropout_prediction_model.pkl")

st.title("Dropout Prediction Model")
features = st.text_input("Enter Student Data (comma-separated)")
print(model.classes_)  # Ensure class order is [0,1]

if st.button("Predict"):
    try:
        input_data = np.array([float(x) for x in features.split(",")]).reshape(1, -1)
        proba = model.predict_proba(input_data)[0]  # Get probability scores

        dropout_prob = proba[0] * 100  # Convert to percentage
        graduated_prob = proba[1] * 100

        st.write(f"**Dropout Probability:** {dropout_prob:.2f}%")
        st.write(f"**Graduation Probability:** {graduated_prob:.2f}%")

        # Add threshold-based classification for better insights
        if dropout_prob > 70:
            st.write("🚨 **High Risk of Dropout! Immediate intervention recommended.**")
        elif dropout_prob > 40:
            st.write("⚠️ **Moderate Risk. Needs monitoring.**")
        else:
            st.write("✅ **Low Risk. Likely to graduate.**")

    except ValueError:
        st.error("Invalid input! Please enter numbers correctly.")
