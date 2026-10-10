import streamlit as st
import pandas as pd
import joblib

from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

model_path = (
    project_root / "models" / "credit_card_customer_segmentation_pipeline.joblib"
)

pipeline = joblib.load(model_path)

st.title("Credit Card Customer Segmentation")
st.write("Enter the customer's financial activity to predict their segment.")

with st.form("customer_form"):
    balance = st.number_input("Balance", min_value=0.0, value=1000.0)
    purchases = st.number_input("Purchases", min_value=0.0, value=500.0)
    oneoff_purchases = st.number_input("One-off Purchases", min_value=0.0, value=200.0)
    installments_purchases = st.number_input(
        "Installments Purchases", min_value=0.0, value=300.0
    )
    cash_advance = st.number_input("Cash Advance", min_value=0.0, value=3000.0)

    credit_limit = st.number_input(
        "Credit Limit", min_value=0.0, value=3000.0
    )

    payments = st.number_input("Payments", min_value=0.0, value=600.0)

    submitted = st.form_submit_button("Predict Customer Segment")

if submitted:
    customer = pd.DataFrame(
        [
            {
                "BALANCE": balance,
                "PURCHASES": purchases,
                "ONEOFF_PURCHASES": oneoff_purchases,
                "INSTALLMENTS_PURCHASES": installments_purchases,
                "CASH_ADVANCE": cash_advance,
                "CREDIT_LIMIT": credit_limit,
                "PAYMENTS": payments,
            }
        ]
    )

    prediction = pipeline.predict(customer)[0]
    st.success(f"Predicted Segment: {prediction}")
