#!/bin/bash

# Quick Start Script cho IoT DDoS Detection với LSTM
# Script này tự động setup môi trường và chạy training

set -e  # Exit on error

echo "=========================================="
echo "IoT DDoS Detection - Quick Start"
echo "=========================================="

# Kiểm tra Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 không được cài đặt"
    exit 1
fi

echo "[1/5] Tạo virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "✓ Đã tạo virtual environment"
else
    echo "✓ Virtual environment đã tồn tại"
fi

echo ""
echo "[2/5] Kích hoạt virtual environment..."
source .venv/bin/activate
echo "✓ Đã kích hoạt virtual environment"

echo ""
echo "[3/5] Cài đặt dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "✓ Đã cài đặt tất cả dependencies"

echo ""
echo "[4/5] Kiểm tra cấu trúc thư mục..."
mkdir -p data/raw data/processed models notebooks results logs
echo "✓ Đã tạo các thư mục cần thiết"

echo ""
echo "[5/5] Kiểm tra dữ liệu..."
if [ -f "data/raw/bot_iot.csv" ]; then
    echo "✓ Đã tìm thấy dữ liệu Bot-IoT"

    echo ""
    echo "=========================================="
    echo "Bắt đầu training với config mặc định?"
    echo "=========================================="
    read -p "Nhấn Enter để tiếp tục hoặc Ctrl+C để thoát..."

    echo ""
    echo "Đang training model..."
    python src/train_lstm.py --config default --data data/raw/bot_iot.csv

    echo ""
    echo "=========================================="
    echo "HOÀN THÀNH!"
    echo "=========================================="
    echo "Model đã được lưu vào: models/lstm_ddos_model.h5"
    echo "Kết quả đánh giá: results/"
    echo "TensorBoard logs: logs/"
    echo ""
    echo "Xem TensorBoard bằng: tensorboard --logdir logs/"

else
    echo "⚠ Không tìm thấy dữ liệu Bot-IoT tại data/raw/bot_iot.csv"
    echo ""
    echo "Vui lòng:"
    echo "  1. Download Bot-IoT dataset từ UNSW Canberra"
    echo "  2. Đặt file CSV vào data/raw/bot_iot.csv"
    echo "  3. Chạy lại script này"
    echo ""
    echo "Hoặc chỉ định đường dẫn khác:"
    echo "  python src/train_lstm.py --config default --data path/to/your/data.csv"
fi

echo ""
echo "Để xem hướng dẫn chi tiết, đọc TRAINING_GUIDE.md"
