import streamlit as st
import pandas as pd
import joblib
from io import BytesIO

# --- App Config ---
st.set_page_config(page_title="FraudSight AI", layout="wide")
st.title("🔍 FraudSight AI – Real-Time Fraud Detection for Financial Transactions")

# --- Upload File ---
uploaded_file = st.file_uploader("📤 Upload your transaction CSV", type=["csv"])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    # Drop unnecessary columns
    drop_cols = ['transaction_id', 'customer_id', 'ip_address', 'device_id']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df.drop('timestamp', axis=1, inplace=True)

    # Encode categorical values
    cat_cols = ['transaction_type', 'channel', 'location', 'customer_status', 'fraud_priority']
    for col in cat_cols:
        df[col] = pd.factorize(df[col])[0]

    # Load model
    model = joblib.load("fraud_model_v2.pkl")

    # Prepare for prediction
    X = df.drop(columns=["is_fraud"], errors="ignore")
    df["fraud_prediction"] = model.predict(X)

    # Add reasons (top 3 features per model)
    feature_importance = pd.Series(model.feature_importances_, index=X.columns)
    top_features = feature_importance.sort_values(ascending=False).head(3).index.tolist()

    def reason_generator(row):
        reasons = []
        for feature in top_features:
            value = row[feature]
            if feature == "amount" and value > 300000:
                reasons.append("High amount")
            elif feature == "hour" and value in [0, 1, 2, 3, 4, 5]:
                reasons.append("Unusual time")
            elif feature == "customer_status" and value == 2:  # assuming 2 = Dormant
                reasons.append("Dormant account")
            elif feature == "balance_after" and value < 1000:
                reasons.append("Very low balance left")
        return ", ".join(reasons[:3]) if reasons else "Unusual pattern"

    df["flag_reason"] = df.apply(lambda row: reason_generator(row) if row["fraud_prediction"] == 1 else "", axis=1)

    # --- Summary Metrics ---
    st.markdown("Fraud Detection Summary")
    total = len(df)
    frauds = df["fraud_prediction"].sum()
    percent = (frauds / total) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", f"{total}")
    col2.metric("Flagged as Fraud", f"{int(frauds)}")
    col3.metric("Fraud Rate", f"{percent:.2f}%")

    # --- Flagged Transactions ---
    st.markdown("### ⚠️ Flagged Transactions with Reasons")
    flagged_df = df[df["fraud_prediction"] == 1].copy()
    if not flagged_df.empty:
        st.dataframe(flagged_df[["amount", "transaction_type", "channel", "customer_status", "hour", "fraud_priority", "flag_reason"]].reset_index(drop=True))

        # Download button
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df_to_csv(flagged_df)
        st.download_button(
            label="📥 Download Fraud Report",
            data=csv,
            file_name='flagged_transactions_report.csv',
            mime='text/csv',
        )
    else:
        st.info("🎉 No fraud detected in this batch.")
else:
    st.info("👈 Please upload a transaction CSV file to begin analysis.")

