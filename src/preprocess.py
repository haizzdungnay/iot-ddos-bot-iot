import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def preprocess_bot_iot(df: pd.DataFrame, feature_cols, label_col):
    X = df[feature_cols].values.astype("float32")
    y = df[label_col].values.astype("int32")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler
