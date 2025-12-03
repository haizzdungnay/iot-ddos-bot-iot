"""
Module đánh giá và visualize kết quả model LSTM
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve,
    average_precision_score
)
from tensorflow.keras.models import load_model
import os
import json

from preprocess import load_preprocessor, reshape_for_lstm


def plot_confusion_matrix(cm, save_path=None):
    """
    Vẽ confusion matrix

    Args:
        cm: confusion matrix
        save_path: đường dẫn lưu figure (nếu None thì chỉ show)
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'])
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu confusion matrix vào {save_path}")
    else:
        plt.show()

    plt.close()


def plot_roc_curve(y_true, y_pred_prob, save_path=None):
    """
    Vẽ ROC curve

    Args:
        y_true: nhãn thật
        y_pred_prob: xác suất dự đoán
        save_path: đường dẫn lưu figure
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=16)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu ROC curve vào {save_path}")
    else:
        plt.show()

    plt.close()

    return roc_auc


def plot_precision_recall_curve(y_true, y_pred_prob, save_path=None):
    """
    Vẽ Precision-Recall curve

    Args:
        y_true: nhãn thật
        y_pred_prob: xác suất dự đoán
        save_path: đường dẫn lưu figure
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob)
    avg_precision = average_precision_score(y_true, y_pred_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AP = {avg_precision:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=16)
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu PR curve vào {save_path}")
    else:
        plt.show()

    plt.close()

    return avg_precision


def plot_training_history(history_path, save_dir=None):
    """
    Vẽ lịch sử training (loss và accuracy)

    Args:
        history_path: đường dẫn file training_history.json
        save_dir: thư mục lưu figures
    """
    with open(history_path, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['loss']) + 1)

    # Plot loss
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    plt.title('Model Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    plt.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    plt.title('Model Accuracy', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    if save_dir:
        save_path = os.path.join(save_dir, 'training_history.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu training history vào {save_path}")
    else:
        plt.show()

    plt.close()


def evaluate_model(model_path, X_test, y_test, output_dir='results'):
    """
    Đánh giá model và tạo báo cáo đầy đủ

    Args:
        model_path: đường dẫn model đã lưu
        X_test: test features (đã reshape cho LSTM)
        y_test: test labels
        output_dir: thư mục lưu kết quả

    Returns:
        dict chứa các metrics
    """
    print("="*80)
    print("ĐÁNH GIÁ MÔ HÌNH LSTM")
    print("="*80)

    # Load model
    print(f"\nLoading model từ {model_path}...")
    model = load_model(model_path)

    # Prediction
    print("Đang dự đoán trên test set...")
    y_pred_prob = model.predict(X_test).ravel()
    y_pred = (y_pred_prob >= 0.5).astype("int32")

    # Confusion Matrix
    print("\n=== Confusion Matrix ===")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Classification Report
    print("\n=== Classification Report ===")
    report_dict = classification_report(y_test, y_pred, digits=4,
                                       target_names=['Normal', 'Attack'],
                                       output_dict=True)
    report_str = classification_report(y_test, y_pred, digits=4,
                                      target_names=['Normal', 'Attack'])
    print(report_str)

    # ROC AUC
    print("\n=== ROC AUC ===")
    roc_auc = auc(*roc_curve(y_test, y_pred_prob)[:2])
    print(f"ROC AUC Score: {roc_auc:.4f}")

    # Average Precision
    avg_precision = average_precision_score(y_test, y_pred_prob)
    print(f"Average Precision Score: {avg_precision:.4f}")

    # Tạo thư mục output
    os.makedirs(output_dir, exist_ok=True)

    # Vẽ các biểu đồ
    print("\n=== Tạo visualizations ===")
    plot_confusion_matrix(cm, os.path.join(output_dir, 'confusion_matrix.png'))
    plot_roc_curve(y_test, y_pred_prob, os.path.join(output_dir, 'roc_curve.png'))
    plot_precision_recall_curve(y_test, y_pred_prob, os.path.join(output_dir, 'pr_curve.png'))

    # Lưu metrics
    metrics = {
        'confusion_matrix': cm.tolist(),
        'classification_report': report_dict,
        'roc_auc': float(roc_auc),
        'average_precision': float(avg_precision)
    }

    metrics_path = os.path.join(output_dir, 'evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✓ Đã lưu metrics vào {metrics_path}")
    print(f"✓ Đã lưu visualizations vào {output_dir}/")

    return metrics


def load_and_evaluate(model_path='models/lstm_ddos_model.h5',
                     data_path='data/processed/bot_iot_preprocessed.csv',
                     time_steps=1,
                     output_dir='results'):
    """
    Load model và data, sau đó đánh giá

    Args:
        model_path: đường dẫn model
        data_path: đường dẫn dữ liệu đã preprocessing
        time_steps: số time steps (phải giống khi train)
        output_dir: thư mục lưu kết quả

    Returns:
        metrics dict
    """
    # Load data
    print(f"Loading dữ liệu từ {data_path}...")
    df = pd.read_csv(data_path)

    # Tách features và labels
    feature_cols = [col for col in df.columns if col != 'label']
    X = df[feature_cols].values.astype('float32')
    y = df['label'].values.astype('int32')

    # Reshape cho LSTM
    X_lstm = reshape_for_lstm(X, time_steps=time_steps)

    print(f"Shape: X={X_lstm.shape}, y={y.shape}")

    # Đánh giá (trên toàn bộ data hoặc có thể split thành test)
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X_lstm, y, test_size=0.2,
                                            random_state=42, stratify=y)

    # Evaluate
    metrics = evaluate_model(model_path, X_test, y_test, output_dir)

    return metrics


def main():
    """
    Main function để chạy evaluation
    """
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate LSTM model')
    parser.add_argument('--model', type=str, default='models/lstm_ddos_model.h5',
                       help='Path to saved model')
    parser.add_argument('--data', type=str, default='data/processed/bot_iot_preprocessed.csv',
                       help='Path to processed data')
    parser.add_argument('--time-steps', type=int, default=1,
                       help='Number of time steps (must match training)')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory for results')

    args = parser.parse_args()

    # Evaluate
    metrics = load_and_evaluate(
        model_path=args.model,
        data_path=args.data,
        time_steps=args.time_steps,
        output_dir=args.output
    )

    print("\n" + "="*80)
    print("HOÀN THÀNH ĐÁNH GIÁ!")
    print("="*80)


if __name__ == "__main__":
    main()
