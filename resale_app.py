import pandas as pd
import streamlit as st
import joblib

def main():
    st.title("Singapore Resale Flat Price Prediction")

    # Load trained model and columns
    model = joblib.load("resale_price_model.pkl")
    training_data = pd.read_csv("feature_engineered_hdb_data.csv")
    training_columns = list(training_data.drop(columns=['resale_price']).columns)

    # User inputs
    town = st.selectbox("Town", options=['ANG MO KIO', 'BEDOK', '...'])
    flat_type = st.selectbox("Flat Type", options=['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE'])
    storey_range = st.selectbox("Storey Range", options=['01 TO 03', '04 TO 06', '07 TO 09', '10 TO 12', '...'])
    floor_area = st.number_input("Floor Area (sqm)", min_value=10.0, max_value=200.0)
    lease_commence_year = st.number_input("Lease Commence Year", min_value=1960, max_value=2024)
    flat_model = st.selectbox("Flat Model", options=['MODEL A', 'MODEL B', '...'])

    if st.button("Predict"):
        remaining_lease = 99 - (2024 - lease_commence_year)
        
        # Create input data with all features
        input_data = {
            'floor_area_sqm': floor_area,
            'remaining_lease': remaining_lease,
            **{f'town_{town}': 1},
            **{f'flat_type_{flat_type}': 1},
            **{f'storey_range_{storey_range}': 1},
            **{f'flat_model_{flat_model}': 1},
        }

        # Ensure all missing columns from training are added with a default value of 0
        for col in training_columns:
            if col not in input_data:
                input_data[col] = 0

        # Align columns and predict
        input_df = pd.DataFrame([input_data])[training_columns]  # Ensure column order matches training
        predicted_price = model.predict(input_df)[0]

        st.success(f"Predicted Resale Price: ${predicted_price:,.2f}")

if __name__ == "__main__":
    main()
