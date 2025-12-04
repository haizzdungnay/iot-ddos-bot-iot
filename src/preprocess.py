"""
Module xử lý dữ liệu Bot-IoT Dataset cho phát hiện DDoS //  preprocess.py
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, List, Optional
import pickle
import os


def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load dữ liệu Bot-IoT từ file CSV

    Args:
        path: đường dẫn tới file CSV

    Returns:
        DataFrame chứa dữ liệu
    """
    print(f"Đang load dữ liệu từ: {path}")
    # low_memory=False để tránh DtypeWarning và đọc chính xác kiểu dữ liệu hơn
    df = pd.read_csv(path, low_memory=False)
    print(f"Đã load {len(df)} mẫu với {len(df.columns)} cột")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Làm sạch dữ liệu: xử lý missing values, duplicates

    Args:
        df: DataFrame gốc

    Returns:
        DataFrame đã được làm sạch
    """
    print("\n=== Làm sạch dữ liệu ===")
    initial_rows = len(df)

    # Xử lý giá trị vô cực
    df = df.replace([np.inf, -np.inf], np.nan)

    # Xử lý missing values
    if df.isnull().sum().sum() > 0:
        print(f"Tìm thấy {df.isnull().sum().sum()} giá trị null")
        df = df.fillna(0)  # hoặc có thể drop: df = df.dropna()

    # Xóa duplicate
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"Tìm thấy {duplicates} bản ghi trùng lặp, đang xóa...")
        df = df.drop_duplicates()

    final_rows = len(df)
    print(f"Đã xóa {initial_rows - final_rows} mẫu. Còn lại: {final_rows} mẫu")

    return df


def encode_labels(
    df: pd.DataFrame,
    label_col: str = 'attack',
    binary: bool = True
) -> Tuple[pd.DataFrame, Optional[LabelEncoder]]:
    """
    Encode nhãn thành dạng số

    Args:
        df: DataFrame
        label_col: tên cột nhãn (vd: 'attack' hoặc 'category')
        binary: True nếu binary classification (Normal=0, Attack=1)

    Returns:
        DataFrame với nhãn đã encode, LabelEncoder object (None nếu binary)
    """
    print(f"\n=== Encoding nhãn từ cột '{label_col}' ===")

    if binary:
        # Trường hợp dùng cột 'attack' với nhãn 0/1 (0 = Normal, 1 = Attack)
        if label_col == "attack":
            # Ép kiểu về int để đảm bảo 0/1
            df["label_binary"] = df[label_col].astype(int)

        # Trường hợp dùng cột 'category': Normal vs các loại tấn công khác
        elif label_col == "category":
            # Normal -> 0, còn lại (DDoS, Reconnaissance, Theft, ...) -> 1
            df["label_binary"] = (
                df[label_col].astype(str).str.lower() != "normal"
            ).astype(int)

        # Trường hợp khác: fallback đơn giản
        else:
            def _to_binary(x):
                s = str(x).lower()
                if s == "normal" or s == "0":
                    return 0
                return 1

            df["label_binary"] = df[label_col].map(_to_binary).astype(int)

        print("Phân phối nhãn binary:")
        print(df["label_binary"].value_counts())
        return df, None

    else:
        # Multi-class classification
        le = LabelEncoder()
        df["label_encoded"] = le.fit_transform(df[label_col])
        print(f"Các lớp: {le.classes_}")
        print("Phân phối nhãn:")
        print(df["label_encoded"].value_counts())
        return df, le


def select_features(df: pd.DataFrame, exclude_cols: Optional[List[str]] = None) -> List[str]:
    """
    Tự động chọn features (tất cả các cột số, loại trừ các cột nhãn)

    Args:
        df: DataFrame
        exclude_cols: danh sách cột cần loại trừ

    Returns:
        Danh sách tên cột features
    """
    if exclude_cols is None:
        exclude_cols = [
            'attack', 'category', 'subcategory', 'label',
            'label_binary', 'label_encoded'
        ]

    # Lấy tất cả cột số
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Loại trừ các cột không phải feature
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]

    print(f"\n=== Đã chọn {len(feature_cols)} features ===")
    return feature_cols


def preprocess_bot_iot(df: pd.DataFrame,
                       feature_cols: List[str],
                       label_col: str = 'label_binary',
                       scaler: Optional[StandardScaler] = None) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Tiền xử lý dữ liệu Bot-IoT: chuẩn hóa features

    Args:
        df: DataFrame đã clean và encode
        feature_cols: danh sách cột features
        label_col: tên cột nhãn (đã encode)
        scaler: StandardScaler đã fit (dùng cho test set), None nếu fit mới

    Returns:
        X_scaled: features đã chuẩn hóa
        y: nhãn
        scaler: StandardScaler object
    """
    print(f"\n=== Tiền xử lý dữ liệu ===")

    X = df[feature_cols].values.astype("float32")
    y = df[label_col].values.astype("int32")

    print(f"Shape X: {X.shape}, Shape y: {y.shape}")

    # Chuẩn hóa
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print("Đã fit và transform StandardScaler")
    else:
        X_scaled = scaler.transform(X)
        print("Đã transform với scaler có sẵn")

    return X_scaled, y, scaler


def reshape_for_lstm(X: np.ndarray, time_steps: int = 1) -> np.ndarray:
    """
    Reshape dữ liệu để phù hợp với input LSTM: (samples, time_steps, features)

    Args:
        X: mảng 2D (samples, features)
        time_steps: số time steps (mặc định 1 cho stateless LSTM)

    Returns:
        Mảng 3D (samples, time_steps, features)
    """
    N, F = X.shape
    X_lstm = X.reshape((N, time_steps, F))
    print(f"Đã reshape từ {X.shape} thành {X_lstm.shape} cho LSTM")
    return X_lstm


def save_preprocessor(scaler: StandardScaler, feature_cols: List[str],
                      output_dir: str = "models") -> None:
    """
    Lưu scaler và danh sách features để dùng khi inference

    Args:
        scaler: StandardScaler đã fit
        feature_cols: danh sách features
        output_dir: thư mục lưu
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open(f"{output_dir}/feature_cols.pkl", "wb") as f:
        pickle.dump(feature_cols, f)

    print(f"\nĐã lưu scaler và feature_cols vào {output_dir}/")


def load_preprocessor(model_dir: str = "models") -> Tuple[StandardScaler, List[str]]:
    """
    Load scaler và danh sách features

    Args:
        model_dir: thư mục chứa scaler

    Returns:
        scaler, feature_cols
    """
    with open(f"{model_dir}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open(f"{model_dir}/feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)

    print(f"Đã load scaler và {len(feature_cols)} features từ {model_dir}/")
    return scaler, feature_cols


def full_preprocessing_pipeline(data_path: str,
                                binary_classification: bool = True,
                                save_processed: bool = True) -> Tuple:
    """
    Pipeline đầy đủ: load → clean → encode → select features → preprocess

    Args:
        data_path: đường dẫn file CSV gốc
        binary_classification: True cho binary, False cho multi-class
        save_processed: có lưu dữ liệu đã xử lý không

    Returns:
        X_scaled, y, scaler, feature_cols
    """
    # 1. Load
    df = load_raw_data(data_path)

    # 2. Clean
    df = clean_data(df)

    # 3. Encode labels
    # Mặc định dùng cột 'attack' (0 = Normal, 1 = Attack).
    # Nếu sau này muốn dùng 'category' thì gọi encode_labels với label_col='category'.
    df, label_encoder = encode_labels(
        df,
        label_col='attack',
        binary=binary_classification
    )
    label_col = 'label_binary' if binary_classification else 'label_encoded'

    # 4. Select features
    feature_cols = select_features(df)

    # 5. Preprocess
    X_scaled, y, scaler = preprocess_bot_iot(df, feature_cols, label_col)

    # 6. Lưu nếu cần
    if save_processed:
        output_path = "data/processed/bot_iot_preprocessed.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        df_processed = pd.DataFrame(X_scaled, columns=feature_cols)
        df_processed['label'] = y
        df_processed.to_csv(output_path, index=False)
        print(f"\nĐã lưu dữ liệu đã xử lý vào {output_path}")

    return X_scaled, y, scaler, feature_cols
