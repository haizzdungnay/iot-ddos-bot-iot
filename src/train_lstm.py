"""
Module huấn luyện mô hình LSTM cho phát hiện DDoS
"""

import numpy as np
import os
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from preprocess import (
    full_preprocessing_pipeline,
    reshape_for_lstm,
    save_preprocessor
)
from config import TrainingConfig, get_config


def build_lstm_model(time_steps: int, num_features: int, config: TrainingConfig):
    """
    Xây dựng mô hình LSTM

    Args:
        time_steps: số time steps
        num_features: số features
        config: TrainingConfig object

    Returns:
        model: Keras model đã compile
    """
    model = Sequential([
        Input(shape=(time_steps, num_features)),
        LSTM(config.model.lstm_units, return_sequences=False),
        Dropout(config.model.dropout_rate),
        Dense(config.model.dense_units, activation="relu"),
        Dropout(config.model.dropout_rate),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.model.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    print("\n=== Kiến trúc mô hình LSTM ===")
    model.summary()

    return model


def build_bidirectional_lstm_model(time_steps: int, num_features: int, config: TrainingConfig):
    """
    Xây dựng mô hình Bidirectional LSTM (cho sequence data)

    Args:
        time_steps: số time steps
        num_features: số features
        config: TrainingConfig object

    Returns:
        model: Keras model đã compile
    """
    model = Sequential([
        Input(shape=(time_steps, num_features)),
        Bidirectional(LSTM(config.model.lstm_units, return_sequences=False)),
        Dropout(config.model.dropout_rate),
        Dense(config.model.dense_units, activation="relu"),
        Dropout(config.model.dropout_rate),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.model.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    print("\n=== Kiến trúc mô hình Bidirectional LSTM ===")
    model.summary()

    return model


def prepare_callbacks(config: TrainingConfig):
    """
    Chuẩn bị callbacks cho training

    Args:
        config: TrainingConfig object

    Returns:
        List of callbacks
    """
    callbacks = []

    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=config.model.early_stopping_patience,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)

    # Model checkpoint
    model_path = os.path.join(config.model.model_dir, config.model.model_name)
    checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_loss',
        save_best_only=config.model.save_best_only,
        verbose=1
    )
    callbacks.append(checkpoint)

    # Reduce learning rate on plateau
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    callbacks.append(reduce_lr)

    # TensorBoard
    log_dir = os.path.join(config.log_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks.append(tensorboard)

    return callbacks


def calculate_class_weights(y_train):
    """
    Tính class weights để xử lý imbalanced data

    Args:
        y_train: nhãn training set

    Returns:
        class_weight dict
    """
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight = dict(zip(classes, weights))

    print(f"\n=== Class weights ===")
    print(f"Class 0 (Normal): {class_weight[0]:.4f}")
    print(f"Class 1 (Attack): {class_weight[1]:.4f}")

    return class_weight


def apply_smote(X_train, y_train):
    """
    Áp dụng SMOTE để balance training data

    Args:
        X_train: training features (2D hoặc 3D)
        y_train: training labels

    Returns:
        X_resampled, y_resampled
    """
    print("\n=== Áp dụng SMOTE ===")

    original_shape = X_train.shape
    is_3d = len(original_shape) == 3

    # SMOTE chỉ làm việc với 2D, nên reshape nếu cần
    if is_3d:
        N, T, F = original_shape
        X_train_2d = X_train.reshape(N, T * F)
    else:
        X_train_2d = X_train

    # Áp dụng SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train_2d, y_train)

    # Reshape lại nếu cần
    if is_3d:
        X_resampled = X_resampled.reshape(-1, T, F)

    print(f"Original: {original_shape} → Resampled: {X_resampled.shape}")
    print(f"Class distribution: {np.bincount(y_resampled)}")

    return X_resampled, y_resampled


def save_training_history(history, config: TrainingConfig):
    """
    Lưu lịch sử training

    Args:
        history: History object từ model.fit
        config: TrainingConfig object
    """
    history_path = os.path.join(config.results_dir, "training_history.json")

    history_dict = {
        'loss': [float(x) for x in history.history['loss']],
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']]
    }

    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)

    print(f"\nĐã lưu training history vào {history_path}")


def train_model(config_name: str = "default", data_path: str = None):
    """
    Hàm chính để huấn luyện model

    Args:
        config_name: tên config ("default", "lightweight", "deep", "sequence")
        data_path: đường dẫn file dữ liệu gốc (nếu None, dùng từ config)

    Returns:
        model, history, metrics
    """
    print("="*80)
    print(f"BẮT ĐẦU TRAINING LSTM - Config: {config_name}")
    print("="*80)

    # 1. Load config
    config = get_config(config_name)

    if data_path is None:
        data_path = config.data.raw_data_path

    # 2. Preprocessing
    print("\n[1/6] Preprocessing dữ liệu...")
    X_scaled, y, scaler, feature_cols = full_preprocessing_pipeline(
        data_path,
        binary_classification=config.data.binary_classification,
        save_processed=True
    )

    # 3. Reshape cho LSTM
    print("\n[2/6] Reshape dữ liệu cho LSTM...")
    X_lstm = reshape_for_lstm(X_scaled, time_steps=config.model.time_steps)

    # 4. Train-test split
    print("\n[3/6] Chia train/test set...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_lstm, y,
        test_size=config.data.test_size,
        random_state=config.data.random_state,
        stratify=y
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # 5. Handle class imbalance
    class_weight = None
    if config.model.use_smote:
        print("\n[4/6] Xử lý imbalanced data với SMOTE...")
        X_train, y_train = apply_smote(X_train, y_train)
    elif config.model.use_class_weight:
        print("\n[4/6] Tính class weights...")
        class_weight = calculate_class_weights(y_train)
    else:
        print("\n[4/6] Không xử lý class imbalance")

    # 6. Build model
    print("\n[5/6] Xây dựng mô hình...")
    _, T, F = X_train.shape

    # Dùng Bidirectional LSTM nếu time_steps > 1
    if config.model.time_steps > 1:
        model = build_bidirectional_lstm_model(T, F, config)
    else:
        model = build_lstm_model(T, F, config)

    # 7. Training
    print("\n[6/6] Bắt đầu training...")
    callbacks = prepare_callbacks(config)

    history = model.fit(
        X_train, y_train,
        epochs=config.model.epochs,
        batch_size=config.model.batch_size,
        validation_split=config.data.validation_split,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )

    # 8. Evaluation
    print("\n" + "="*80)
    print("ĐÁNH GIÁ MÔ HÌNH")
    print("="*80)

    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob >= 0.5).astype("int32").ravel()

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, digits=4,
                                   target_names=['Normal', 'Attack'])
    print(report)

    # 9. Save results
    save_training_history(history, config)
    save_preprocessor(scaler, feature_cols, config.model.model_dir)

    # Lưu metrics
    metrics = {
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(y_test, y_pred,
                                                       output_dict=True)
    }

    metrics_path = os.path.join(config.results_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✓ Model đã lưu vào: {config.model.model_dir}/{config.model.model_name}")
    print(f"✓ Metrics đã lưu vào: {metrics_path}")
    print(f"✓ TensorBoard logs: {config.log_dir}")

    return model, history, metrics


def main():
    """
    Hàm main - có thể chỉnh sửa config và data path ở đây
    """
    import argparse

    parser = argparse.ArgumentParser(description='Train LSTM model for DDoS detection')
    parser.add_argument('--config', type=str, default='default',
                       choices=['default', 'lightweight', 'deep', 'sequence'],
                       help='Config name')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to raw data CSV file')

    args = parser.parse_args()

    # Train model
    model, history, metrics = train_model(
        config_name=args.config,
        data_path=args.data
    )

    print("\n" + "="*80)
    print("HOÀN THÀNH!")
    print("="*80)


if __name__ == "__main__":
    main()
