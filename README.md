# Phát hiện DDoS IoT với mô hình LSTM trên Bot-IoT Dataset

## 1. Mục tiêu đề tài (phần LSTM)

Trong phạm vi dự án này, nội dung công việc tập trung **chỉ vào mô hình LSTM**, bao gồm:

- Xây dựng hệ thống phát hiện tấn công DDoS trên mạng IoT sử dụng mô hình LSTM.
- Sử dụng Bot-IoT dataset làm dữ liệu huấn luyện và đánh giá.
- Xây dựng pipeline hoàn chỉnh: tải dữ liệu → tiền xử lý → chuẩn hóa → chuẩn bị input cho LSTM → huấn luyện → đánh giá kết quả.
- Đánh giá mô hình LSTM theo các chỉ số: accuracy, precision, recall, F1-score, confusion matrix, tập trung vào khả năng phát hiện lớp tấn công DDoS.

Các mô hình khác như CNN, Hybrid CNN–LSTM **không nằm trong phạm vi thực hiện** của dự án này.

---

## 2. Dataset: Bot-IoT

- Nguồn: Bot-IoT dataset (UNSW Canberra).
- Dạng dữ liệu sử dụng: file CSV chứa các flow/connection mạng, có nhãn normal/attack (hoặc normal/DDoS).
- Phạm vi sử dụng:
  - Giải bài toán phân loại nhị phân: Normal vs DDoS (hoặc Normal vs Attack sau khi lọc lại nhãn).
  - Dữ liệu dùng để huấn luyện và đánh giá mô hình LSTM, không triển khai trên thiết bị IoT thực.

---

## 3. Cấu trúc thư mục project

```text
iot-ddos-bot-iot/
├─ .venv/                     # virtual environment (Python)
├─ data/
│  ├─ raw/                    # Bot-IoT gốc (CSV)
│  └─ processed/              # dữ liệu sau tiền xử lý (CSV đã làm sạch, chuẩn hóa)
├─ models/                    # lưu model LSTM đã huấn luyện (.h5, .pkl)
├─ notebooks/                 # Jupyter notebooks cho demo và experiments
│  └─ demo_training.ipynb     # Demo notebook
├─ results/                   # kết quả đánh giá (metrics, plots)
├─ logs/                      # TensorBoard logs
├─ src/
│  ├─ __init__.py
│  ├─ config.py               # quản lý cấu hình training
│  ├─ preprocess.py           # load + tiền xử lý dữ liệu Bot-IoT
│  ├─ train_lstm.py           # huấn luyện mô hình LSTM
│  └─ evaluate.py             # đánh giá và visualize kết quả
├─ requirements.txt           # danh sách thư viện Python
├─ README.md                  # mô tả đề tài (file này)
├─ TRAINING_GUIDE.md          # hướng dẫn training chi tiết
└─ .gitignore                 # git ignore
```

---

## 4. Hướng dẫn sử dụng

### 4.1. Cài đặt môi trường

```bash
# Clone repository (nếu dùng git)
git clone <repository-url>
cd iot-ddos-bot-iot

# Tạo virtual environment
python -m venv .venv

# Kích hoạt virtual environment
source .venv/bin/activate  # Linux/Mac
# hoặc: .venv\Scripts\activate  # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

### 4.2. Chuẩn bị dữ liệu

1. Download Bot-IoT dataset từ [UNSW Canberra](https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/)
2. Đặt file CSV vào thư mục `data/raw/`:
   ```bash
   cp bot_iot.csv data/raw/bot_iot.csv
   ```

### 4.3. Training Model

#### **Cách 1: Command Line (Đề xuất)**

```bash
# Training với config mặc định
python src/train_lstm.py --config default --data data/raw/bot_iot.csv

# Training với config khác
python src/train_lstm.py --config lightweight --data data/raw/bot_iot.csv
python src/train_lstm.py --config deep --data data/raw/bot_iot.csv
```

**Các config có sẵn**:
- `default`: Config cân bằng (64 LSTM units, 50 epochs)
- `lightweight`: Model nhẹ, training nhanh (32 units, 20 epochs)
- `deep`: Model sâu hơn (128 units, 100 epochs)
- `sequence`: Sử dụng sequence LSTM với time_steps=10

#### **Cách 2: Jupyter Notebook**

```bash
jupyter notebook notebooks/demo_training.ipynb
```

Mở notebook và chạy từng cell theo hướng dẫn.

#### **Cách 3: Python Script**

```python
from src.train_lstm import train_model

# Train model
model, history, metrics = train_model(
    config_name='default',
    data_path='data/raw/bot_iot.csv'
)
```

### 4.4. Monitor Training

Sử dụng TensorBoard để theo dõi quá trình training real-time:

```bash
tensorboard --logdir logs/
```

Truy cập: http://localhost:6006

### 4.5. Đánh giá Model

```bash
python src/evaluate.py \
    --model models/lstm_ddos_model.h5 \
    --data data/processed/bot_iot_preprocessed.csv \
    --output results/
```

Kết quả sẽ được lưu trong thư mục `results/`:
- `confusion_matrix.png`: Ma trận nhầm lẫn
- `roc_curve.png`: ROC curve
- `pr_curve.png`: Precision-Recall curve
- `training_history.png`: Lịch sử training
- `evaluation_metrics.json`: Metrics chi tiết

### 4.6. Inference trên dữ liệu mới

```python
import pandas as pd
from tensorflow.keras.models import load_model
from src.preprocess import load_preprocessor, reshape_for_lstm

# Load model và preprocessor
model = load_model('models/lstm_ddos_model.h5')
scaler, feature_cols = load_preprocessor('models')

# Load dữ liệu mới
new_data = pd.read_csv('path_to_new_data.csv')
X_new = new_data[feature_cols].values.astype('float32')

# Preprocess
X_new_scaled = scaler.transform(X_new)
X_new_lstm = reshape_for_lstm(X_new_scaled, time_steps=1)

# Dự đoán
predictions = model.predict(X_new_lstm)
predicted_labels = (predictions >= 0.5).astype('int32').ravel()

# 0: Normal, 1: Attack/DDoS
```

---

## 5. Tài liệu chi tiết

Xem **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** để có:
- Hướng dẫn chi tiết về chiến lược training
- Giải thích các hyperparameters
- Cách xử lý class imbalance
- Troubleshooting và best practices
- Phân tích metrics và tối ưu model

---

## 6. Kết quả mong đợi

Với Bot-IoT dataset và config mặc định, model LSTM nên đạt:

| Metric | Target |
|--------|--------|
| Accuracy | ≥ 95% |
| Precision | ≥ 90% |
| Recall | ≥ 95% |
| F1-Score | ≥ 92% |
| ROC AUC | ≥ 0.95 |

**Lưu ý**: Kết quả thực tế phụ thuộc vào chất lượng dữ liệu và config sử dụng.

---

## 7. Troubleshooting

### Lỗi thường gặp:

**1. FileNotFoundError: data/raw/bot_iot.csv**
- Kiểm tra đường dẫn file CSV
- Đảm bảo file đã được đặt đúng trong `data/raw/`

**2. KeyError: 'attack'**
- Kiểm tra tên cột nhãn trong dataset
- Chỉnh `label_column` trong `src/config.py`

**3. Out of Memory**
- Giảm `batch_size` trong config (256 → 128 hoặc 64)
- Sử dụng config `lightweight`

**4. Model không học (loss không giảm)**
- Giảm `learning_rate` (1e-3 → 5e-4 hoặc 1e-4)
- Kiểm tra data có bị lỗi (NaN, inf)

Xem thêm trong [TRAINING_GUIDE.md](TRAINING_GUIDE.md#troubleshooting)

---

## 8. Tài liệu tham khảo

- **Bot-IoT Dataset**: https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/
- **LSTM Theory**: https://colah.github.io/posts/2015-08-Understanding-LSTMs/
- **TensorFlow/Keras**: https://www.tensorflow.org/guide/keras
- **Imbalanced Learning**: https://imbalanced-learn.org/

---

## 9. License

Dự án này sử dụng Bot-IoT dataset từ UNSW Canberra. Vui lòng tham khảo license của dataset khi sử dụng.

---

## 10. Contact

Nếu có câu hỏi hoặc gặp vấn đề, vui lòng tạo issue trên GitHub repository.