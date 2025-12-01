import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Load Model
model = pickle.load(open("model.pkl", "rb"))

st.title("🐦 Twitter Social Media Influence Prediction System")

st.write("""
This web app predicts **who is more influential on Twitter**  
based on the Kaggle dataset *'Influencers in Social Networks'*.
""")

# File Upload Section
st.header("📤 Upload CSV File")
uploaded = st.file_uploader("Upload train/test CSV", type=['csv'])

if uploaded:
    df = pd.read_csv(uploaded)
    st.subheader("Uploaded Data Preview")
    st.write(df.head())

    # Predict Button
    if st.button("🔍 Predict Influence"):
        try:
            X = df.drop("Choice", axis=1, errors="ignore")  
            predictions = model.predict(X)

            df["Predicted_Influencer"] = predictions
            st.subheader("📌 Prediction Results")
            st.write(df)

            # Download Output
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Error: {e}")

# Graph Section
st.header("📊 Data Visualization")

if st.checkbox("Show Correlation Heatmap"):
    if uploaded:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(), annot=False, cmap="Blues")
        st.pyplot(fig)
    else:
        st.warning("Please upload a CSV first.")

if st.checkbox("Show Feature Importance (XGBoost)"):
    fig, ax = plt.subplots(figsize=(10, 5))
    xgb.plot_importance(model, ax=ax)
    st.pyplot(fig)
