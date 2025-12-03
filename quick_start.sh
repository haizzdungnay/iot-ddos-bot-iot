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
if [ -f "data/raw/bot_iot.csv" ] || ls data/raw/*.csv 1> /dev/null 2>&1; then
    echo "✓ Đã tìm thấy dữ liệu CSV trong data/raw/"

    echo ""
    echo "=========================================="
    echo "Bắt đầu training với config mặc định?"
    echo "=========================================="
    read -p "Nhấn Enter để tiếp tục hoặc Ctrl+C để thoát..."

    echo ""
    echo "Đang training model..."

    # Tìm file CSV đầu tiên
    CSV_FILE=$(ls data/raw/*.csv 2>/dev/null | head -n 1)
    if [ -f "data/raw/bot_iot.csv" ]; then
        CSV_FILE="data/raw/bot_iot.csv"
    fi

    echo "Sử dụng file: $CSV_FILE"
    python src/train_lstm.py --config default --data "$CSV_FILE"

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
    echo "⚠ Không tìm thấy dữ liệu trong data/raw/"
    echo ""
    echo "=========================================="
    echo "HƯỚNG DẪN DOWNLOAD DATASET"
    echo "=========================================="
    echo ""
    echo "Option 1: UNSW OneDrive (Khuyến nghị - Dễ nhất, Chính thức) ⭐"
    echo "  Link: https://unsw-my.sharepoint.com/:f:/g/personal/z5131399_ad_unsw_edu_au/EvWlXAuKAMlOq7alJAV-JG0BbQM-P_j51_tBX_EXlCz6fg"
    echo "  - Nguồn chính thức từ UNSW"
    echo "  - CSV format ~16.7GB, PCAP ~69.3GB"
    echo "  - Click Download trên OneDrive"
    echo ""
    echo "Option 2: Nguồn thay thế"
    echo "  - IMPACT CyberTrust: https://www.impactcybertrust.org/dataset_view?idDataset=1296"
    echo "  - OpenML: https://www.openml.org/d/42072"
    echo ""
    echo "Option 3: Kaggle Alternatives"
    echo "  - CIC-BoT-IoT: https://www.kaggle.com/datasets/dhoogla/cicbotiot"
    echo "  - NF-BoT-IoT: https://www.kaggle.com/datasets/dhoogla/nfbotiot"
    echo ""
    echo "Option 4: Dataset mới hơn"
    echo "  - CIC IoT-DIAD 2024: https://www.unb.ca/cic/datasets/iot-diad-2024.html"
    echo "  - IoT-DH: https://data.mendeley.com/datasets/8dns3xbckv/1"
    echo ""
    echo "📖 Xem hướng dẫn chi tiết: DATASET_SETUP.md"
    echo ""
    echo "Sau khi download, chạy lại script này hoặc:"
    echo "  python src/train_lstm.py --config default --data data/raw/your_file.csv"
fi

echo ""
echo "Để xem hướng dẫn chi tiết, đọc TRAINING_GUIDE.md"
