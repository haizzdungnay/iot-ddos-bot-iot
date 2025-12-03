import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

from preprocess import load_raw_data, preprocess_bot_iot

def build_lstm_model(time_steps: int, num_features: int):
    model = Sequential([
        Input(shape=(time_steps, num_features)),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

def main():
    # TODO: chỉnh lại đường dẫn và tên cột cho đúng Bot-IoT của bạn
    df = load_raw_data("data/processed/bot_iot_preprocessed.csv")
    feature_cols = [...]  # danh sách tên cột feature
    label_col = "label"

    X_scaled, y, scaler = preprocess_bot_iot(df, feature_cols, label_col)

    N, F = X_scaled.shape
    T = 1
    X_lstm = X_scaled.reshape((N, T, F))

    X_train, X_test, y_train, y_test = train_test_split(
        X_lstm, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_lstm_model(T, F)

    model.fit(
        X_train, y_train,
        epochs=5,          # để test pipeline, sau tăng lên
        batch_size=256,
        validation_split=0.1,
        verbose=1
    )

    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob >= 0.5).astype("int32").ravel()

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=4))

if __name__ == "__main__":
    main()
