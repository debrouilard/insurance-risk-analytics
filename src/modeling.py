import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import shap

def prepare_data(df):
    """Feature engineering and data preparation"""
    df = df.copy()
    
    # Feature Engineering
    if 'RegistrationYear' in df.columns:
        df['VehicleAge'] = 2015 - df['RegistrationYear']  # assuming data up to 2015
    
    if 'TransactionMonth' in df.columns:
        df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'])
        df['PolicyDuration'] = (df['TransactionMonth'].max() - df['TransactionMonth']).dt.days
    
    # Select features
    categorical_cols = ['Province', 'Gender', 'VehicleType', 'Make', 'CoverType']
    numerical_cols = ['VehicleAge', 'SumInsured', 'CustomValueEstimate', 'PolicyDuration']
    
    # One-hot encoding
    df_encoded = pd.get_dummies(df[categorical_cols + numerical_cols], drop_first=True)
    
    return df_encoded, df['TotalClaims']

def train_severity_models(X, y):
    """Train three models for claim severity"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        results[name] = {"RMSE": rmse, "R2": r2, "Model": model}
    
    return results, X_test

def explain_model(model, X_test):
    """SHAP explanation for best model"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    return shap_values