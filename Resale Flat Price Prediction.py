#Resale Price Prediction Project

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib

def preprocess_data():
    # Load dataset
    dataset = pd.read_csv("Resale Flat Prices .csv")

    # Clean the data
    data = dataset.dropna()  # Remove rows with missing values

    # Convert data types where necessary
    data['lease_commence_date'] = pd.to_datetime(data['lease_commence_date'], errors='coerce')

    # Handle erroneous data or duplicates
    data = data.drop_duplicates()

    # Save cleaned dataset for feature engineering
    cleaned_data_path = "cleaned_hdb_data.csv"
    data.to_csv(cleaned_data_path, index=False)
    print(f"Cleaned data saved to {cleaned_data_path}")

def feature_engineering():
    # Load cleaned data
    data = pd.read_csv("cleaned_hdb_data.csv")

    # Select and create relevant features
    data['remaining_lease'] = 99 - (2024 - pd.to_datetime(data['lease_commence_date']).dt.year)
    data = data[['town', 'flat_type', 'storey_range', 'floor_area_sqm', 'flat_model', 'remaining_lease', 'resale_price']]

    # Convert categorical features into numerical using one-hot encoding
    data = pd.get_dummies(data, columns=['town', 'flat_type', 'storey_range', 'flat_model'], drop_first=True)

    # Save feature-engineered dataset
    feature_engineered_path = "feature_engineered_hdb_data.csv"
    data.to_csv(feature_engineered_path, index=False)
    print(f"Feature-engineered data saved to {feature_engineered_path}")

def train_model():
    # Load feature-engineered data
    data = pd.read_csv("feature_engineered_hdb_data.csv")

    # Split data into features and target
    X = data.drop('resale_price', axis=1)
    y = data['resale_price']

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, "resale_price_model.pkl")
    print("Model saved as 'resale_price_model.pkl'")

    # Model evaluation
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    print(f"R2 Score: {r2}")

if __name__ == "__main__":
    preprocess_data()
    preprocess_data()
    train_model()
    
