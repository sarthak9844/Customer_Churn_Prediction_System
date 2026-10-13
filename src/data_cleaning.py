import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def clean_data(df):
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"],errors="coerce")
    df = df.dropna(subset=["TotalCharges"])
    df = df.drop("customerID", axis=1)
    return df