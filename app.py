import streamlit as st
import pandas as pd
import joblib

# Page config
st.set_page_config(page_title="FraudSight AI", layout="wide")
st.title("🔍 FraudSight AI – Real-Time Fraud Detection for Financial Transactions")

st.markdown("""
Upload a CSV file of recent transactions to detect fraudulent activity in real-time using our AI model.
""")

# File uploader
uploaded_file = st.file_uploader("📤 Upload your transaction CSV", type=["csv"])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    # Display basic stats
    st.write("### 🧾 Uploaded Data Preview")
    st.dataframe(df.head())

    # Preprocessing
    drop_cols = ['transaction_id', 'customer_id', 'ip_address', 'device_id']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    # Convert timestamp to features
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df.drop('timestamp', axis=1, inplace=True)

    # Encode categorical values
    cat_cols = ['transaction_type', 'channel', 'location', 'customer_status', 'fraud_priority']
    for col in cat_cols:
        df[col] = pd.factorize(df[col])[0]

    # Load model
    model = joblib.load("fraud_model.pkl")

    # Predict
    X = df.drop(columns=["is_fraud"], errors="ignore")
    df["fraud_prediction"] = model.predict(X)

    # Show summary
    total = len(df)
    frauds = df["fraud_prediction"].sum()
    percent = (frauds / total) * 100

    st.markdown("### 📊 Fraud Detection Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", f"{total}")
    col2.metric("Flagged as Fraud", f"{int(frauds)}")
    col3.metric("Fraud Rate", f"{percent:.2f}%")

    # Show frauds
    st.markdown("### ⚠️ Flagged Transactions")
    flagged_df = df[df["fraud_prediction"] == 1].copy()
    if not flagged_df.empty:
        st.dataframe(flagged_df.reset_index(drop=True))
    else:
        st.info("No fraud detected in this batch.")
else:
    st.info("👈 Please upload a transaction CSV file to begin analysis.")
