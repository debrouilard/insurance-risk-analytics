import pandas as pd
import os

def load_data(file_path: str = None) -> pd.DataFrame:
    """Load the insurance dataset with flexible path handling."""
    
    # Default paths - try multiple possible locations
    if file_path is None:
        possible_paths = [
            "data/raw/insurance_data.csv",                    # from root
            "../data/raw/insurance_data.csv",                 # from notebooks folder
            "../../data/raw/insurance_data.csv",              # extra safety
            "data/raw/MachineLearningRating_v3.txt"           # original name fallback
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                file_path = path
                print(f"✅ Found data at: {path}")
                break
        else:
            raise FileNotFoundError("Dataset not found in any expected location.")
    
    print(f"Loading data from: {file_path}")
    
    try:
        df = pd.read_csv(file_path, sep='|', low_memory=False)
        print("✅ Loaded successfully using '|' separator")
    except:
        df = pd.read_csv(file_path, low_memory=False)
        print("✅ Loaded successfully using comma separator")
    
    print(f"✅ Data loaded successfully. Shape: {df.shape}")
    return df

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Basic data cleaning and feature engineering."""
    df = df.copy()
    
    if 'TransactionMonth' in df.columns:
        df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
    
    if 'TotalClaims' in df.columns and 'TotalPremium' in df.columns:
        df['LossRatio'] = df['TotalClaims'] / df['TotalPremium'].replace(0, float('nan'))
        df['Margin'] = df['TotalPremium'] - df['TotalClaims']
    
    df = df.dropna(subset=['TotalPremium', 'TotalClaims'])
    
    print(f"✅ Basic cleaning completed. New shape: {df.shape}")
    return df

if __name__ == "__main__":
    df = load_data()
    df_clean = basic_cleaning(df)
    
    os.makedirs("data/cleaned", exist_ok=True)
    cleaned_path = "data/cleaned/insurance_data_cleaned.csv"
    df_clean.to_csv(cleaned_path, index=False)
    print(f"✅ Cleaned data saved to {cleaned_path}")