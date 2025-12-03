# Hướng dẫn Training Model LSTM cho Phát hiện DDoS

## 📋 Mục lục

1. [Tổng quan](#tổng-quan)
2. [Chuẩn bị dữ liệu](#chuẩn-bị-dữ-liệu)
3. [Chiến lược Training](#chiến-lược-training)
4. [Các Config được đề xuất](#các-config-được-đề-xuất)
5. [Quy trình Training](#quy-trình-training)
6. [Đánh giá và Tối ưu](#đánh-giá-và-tối-ưu)
7. [Troubleshooting](#troubleshooting)

---

## 🎯 Tổng quan

Dự án này sử dụng **LSTM (Long Short-Term Memory)** để phát hiện tấn công DDoS trên mạng IoT với Bot-IoT dataset.

### Đặc điểm của bài toán:

- **Loại**: Binary Classification (Normal vs Attack/DDoS)
- **Dataset**: Bot-IoT (UNSW Canberra)
- **Model**: LSTM / Bidirectional LSTM
- **Thách thức chính**: Class imbalance (thường có nhiều Normal hơn Attack)

---

## 📊 Chuẩn bị dữ liệu

### 1. Download Bot-IoT Dataset

Bot-IoT dataset có sẵn trên nhiều nguồn:

#### **Nguồn Khuyến nghị: Kaggle** ⭐

**Option A: Bot-IoT Full Dataset**
```bash
# Cài đặt Kaggle CLI
pip install kaggle

# Download (cần cấu hình Kaggle API token trước)
kaggle datasets download -d vigneshvenkateswaran/bot-iot -p data/raw/ --unzip
```
- Link: [https://www.kaggle.com/datasets/vigneshvenkateswaran/bot-iot](https://www.kaggle.com/datasets/vigneshvenkateswaran/bot-iot)
- Kích thước: ~16GB (full)

**Option B: Bot-IoT 5% Sample** (Nhẹ hơn, khuyến nghị cho test)
```bash
kaggle datasets download -d vigneshvenkateswaran/bot-iot-5-data -p data/raw/ --unzip
```
- Link: [https://www.kaggle.com/datasets/vigneshvenkateswaran/bot-iot-5-data](https://www.kaggle.com/datasets/vigneshvenkateswaran/bot-iot-5-data)
- Kích thước: ~800MB

**Download thủ công từ Kaggle:**
1. Truy cập link trên
2. Đăng nhập Kaggle (miễn phí)
3. Click "Download"
4. Giải nén vào `data/raw/`

#### **Nguồn thay thế:**

- **CIC IoT-DIAD 2024**: [https://www.unb.ca/cic/datasets/iot-diad-2024.html](https://www.unb.ca/cic/datasets/iot-diad-2024.html)
- **CICIoT2023**: [https://www.unb.ca/cic/datasets/iotdataset-2023.html](https://www.unb.ca/cic/datasets/iotdataset-2023.html)
- **IoT-DH Dataset**: [https://data.mendeley.com/datasets/8dns3xbckv/1](https://data.mendeley.com/datasets/8dns3xbckv/1)

### 2. Đặt dữ liệu vào thư mục

```bash
# Nếu file có tên khác, đổi tên:
mv data/raw/UNSW_2018_IoT_Botnet_Dataset*.csv data/raw/bot_iot.csv

# Hoặc dùng trực tiếp với --data flag:
python src/train_lstm.py --config default --data data/raw/UNSW_2018_IoT_Botnet_Dataset_5.csv
```

### 3. Kiểm tra dữ liệu

Dữ liệu Bot-IoT cần có:
- **Cột nhãn**: Thường là `attack`, `category`, hoặc tương tự
  - Giá trị: "Normal", "DDoS", "DoS", v.v.
- **Cột features**: Các đặc trưng mạng (flow duration, packet size, protocol, v.v.)

**Lưu ý**: Nếu tên cột nhãn khác, cần chỉnh trong `config.py` → `DataConfig.label_column`

---

## 🎓 Chiến lược Training

### 1. Xử lý Class Imbalance

Bot-IoT thường có **imbalance** giữa Normal và Attack. Có 3 cách xử lý:

#### **Cách 1: Class Weights (Đề xuất)**
- Tự động tính trọng số cho từng class
- Không tăng kích thước dataset
- **Khi nào dùng**: Mặc định, phù hợp với hầu hết trường hợp

```python
# Trong config.py
use_class_weight = True
use_smote = False
```

#### **Cách 2: SMOTE (Synthetic Minority Over-sampling)**
- Tạo thêm dữ liệu synthetic cho class thiểu số
- Tăng kích thước training set
- **Khi nào dùng**: Class imbalance rất nặng (tỷ lệ > 1:10)

```python
# Trong config.py
use_class_weight = False
use_smote = True
```

#### **Cách 3: Không xử lý**
- Chỉ dùng khi data đã balanced
- **Khi nào dùng**: Sau khi đã undersample/oversample thủ công

```python
# Trong config.py
use_class_weight = False
use_smote = False
```

### 2. Chọn kiến trúc LSTM

#### **Stateless LSTM (time_steps=1)**
- Mỗi mẫu là 1 network flow độc lập
- **Ưu điểm**: Đơn giản, nhanh, phù hợp với Bot-IoT
- **Nhược điểm**: Không khai thác temporal dependency

```python
# Trong config.py
time_steps = 1
```

#### **Sequence LSTM (time_steps>1)**
- Nhóm nhiều flows liên tiếp thành sequence
- **Ưu điểm**: Khai thác temporal patterns
- **Nhược điểm**: Cần nhiều data hơn, phức tạp hơn

```python
# Trong config.py
time_steps = 10  # Ví dụ: 10 flows liên tiếp
```

**Khuyến nghị**: Bắt đầu với `time_steps=1`, sau đó thử nghiệm với sequence nếu cần.

### 3. Hyperparameters chính

| Parameter | Ý nghĩa | Giá trị đề xuất |
|-----------|---------|-----------------|
| `lstm_units` | Số units trong LSTM layer | 64 (default), 32 (light), 128 (deep) |
| `dropout_rate` | Tỷ lệ dropout (chống overfit) | 0.3 - 0.4 |
| `dense_units` | Số units trong Dense layer | 32 (default), 16 (light), 64 (deep) |
| `learning_rate` | Learning rate cho Adam optimizer | 1e-3 (0.001) |
| `batch_size` | Batch size | 256 - 512 |
| `epochs` | Số epochs tối đa | 50 - 100 |
| `early_stopping_patience` | Patience cho early stopping | 10 - 15 |

---

## ⚙️ Các Config được đề xuất

Dự án cung cấp 4 config có sẵn:

### 1. **Default Config** (Khuyến nghị bắt đầu)

```python
# Chạy với:
python src/train_lstm.py --config default

# Hoặc trong code:
from train_lstm import train_model
model, history, metrics = train_model(config_name='default')
```

**Đặc điểm**:
- LSTM units: 64
- Dense units: 32
- Dropout: 0.3
- Epochs: 50
- Batch size: 256
- Class weight: True

**Khi nào dùng**: Bắt đầu experiment, baseline model

### 2. **Lightweight Config** (Nhanh, nhẹ)

```python
python src/train_lstm.py --config lightweight
```

**Đặc điểm**:
- LSTM units: 32 (giảm)
- Dense units: 16 (giảm)
- Batch size: 512 (tăng)
- Epochs: 20 (giảm)

**Khi nào dùng**:
- Testing pipeline nhanh
- Dataset nhỏ
- Tài nguyên hạn chế

### 3. **Deep Config** (Model sâu hơn)

```python
python src/train_lstm.py --config deep
```

**Đặc điểm**:
- LSTM units: 128 (tăng)
- Dense units: 64 (tăng)
- Dropout: 0.4 (tăng)
- Epochs: 100 (tăng)
- Early stopping patience: 15

**Khi nào dùng**:
- Dataset lớn (>100K samples)
- Muốn maximize performance
- Có GPU mạnh

### 4. **Sequence Config** (Sequence LSTM)

```python
python src/train_lstm.py --config sequence
```

**Đặc điểm**:
- Time steps: 10 (sử dụng sequence)
- LSTM units: 128
- Epochs: 100
- Sử dụng Bidirectional LSTM

**Khi nào dùng**:
- Data có temporal dependency
- Muốn thử nghiệm sequence modeling
- **Lưu ý**: Cần xử lý data khác (group flows theo time)

---

## 🚀 Quy trình Training

### Bước 1: Cài đặt môi trường

```bash
# Tạo virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# hoặc .venv\Scripts\activate  # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

### Bước 2: Chuẩn bị dữ liệu

Đặt file Bot-IoT CSV vào `data/raw/bot_iot.csv`

### Bước 3: Chạy training

#### **Option 1: Command line (Đề xuất)**

```bash
# Training với default config
python src/train_lstm.py --config default --data data/raw/bot_iot.csv

# Hoặc với config khác
python src/train_lstm.py --config lightweight --data data/raw/bot_iot.csv
```

#### **Option 2: Jupyter Notebook**

```bash
jupyter notebook notebooks/demo_training.ipynb
```

Mở notebook và chạy từng cell theo thứ tự.

#### **Option 3: Python script**

```python
from train_lstm import train_model

# Train
model, history, metrics = train_model(
    config_name='default',
    data_path='data/raw/bot_iot.csv'
)
```

### Bước 4: Monitor training

#### **TensorBoard**

```bash
# Mở TensorBoard để theo dõi real-time
tensorboard --logdir logs/
```

Truy cập: http://localhost:6006

#### **Output trong console**

Training sẽ hiển thị:
- Progress bar từng epoch
- Loss, accuracy, precision, recall
- Validation metrics
- Early stopping notifications

### Bước 5: Đánh giá kết quả

Sau khi training, các file sẽ được tạo:

```
models/
  ├── lstm_ddos_model.h5      # Model đã train
  ├── scaler.pkl              # StandardScaler
  └── feature_cols.pkl        # Danh sách features

results/
  ├── training_history.json   # Lịch sử training
  └── metrics.json            # Metrics đánh giá

logs/
  └── <timestamp>/            # TensorBoard logs
```

#### Đánh giá chi tiết:

```bash
python src/evaluate.py \
    --model models/lstm_ddos_model.h5 \
    --data data/processed/bot_iot_preprocessed.csv \
    --output results/
```

Kết quả:
```
results/
  ├── confusion_matrix.png
  ├── roc_curve.png
  ├── pr_curve.png
  ├── training_history.png
  └── evaluation_metrics.json
```

---

## 📈 Đánh giá và Tối ưu

### 1. Metrics quan trọng

#### **Confusion Matrix**
```
                Predicted
              Normal  Attack
Actual Normal   TN      FP
       Attack   FN      TP
```

- **True Negative (TN)**: Normal được phát hiện đúng
- **False Positive (FP)**: Normal bị nhầm thành Attack (False Alarm)
- **False Negative (FN)**: Attack bị bỏ sót (Nguy hiểm!)
- **True Positive (TP)**: Attack được phát hiện đúng

#### **Chỉ số đánh giá**

| Metric | Công thức | Ý nghĩa | Mục tiêu |
|--------|-----------|---------|----------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Tỷ lệ dự đoán đúng tổng thể | >95% |
| **Precision** | TP/(TP+FP) | Trong các mẫu dự đoán Attack, bao nhiêu đúng? | >90% |
| **Recall** | TP/(TP+FN) | Trong các Attack thật, phát hiện được bao nhiêu? | **>95%** (quan trọng nhất) |
| **F1-Score** | 2×(Precision×Recall)/(Precision+Recall) | Cân bằng giữa Precision và Recall | >92% |
| **ROC AUC** | Area Under ROC Curve | Khả năng phân biệt 2 class | >0.95 |

**Lưu ý**: Với DDoS detection, **Recall** (detect được bao nhiêu attack) quan trọng hơn Precision (giảm false alarm).

### 2. Phân tích kết quả

#### **Trường hợp 1: High Accuracy, Low Recall**
- **Nguyên nhân**: Model bias về class Normal
- **Giải pháp**:
  - Tăng class weight cho Attack
  - Sử dụng SMOTE
  - Điều chỉnh threshold (từ 0.5 → 0.3)

#### **Trường hợp 2: Overfitting (train acc >> val acc)**
- **Nguyên nhân**: Model học thuộc training data
- **Giải pháp**:
  - Tăng dropout rate (0.3 → 0.4 - 0.5)
  - Giảm số LSTM units
  - Thêm regularization (L2)
  - Tăng training data

#### **Trường hợp 3: Underfitting (train acc thấp)**
- **Nguyên nhân**: Model quá đơn giản
- **Giải pháp**:
  - Tăng số LSTM units (64 → 128)
  - Thêm LSTM layers
  - Giảm dropout
  - Train lâu hơn

### 3. Hyperparameter Tuning

#### **Grid Search thủ công**

Thử nghiệm các kết hợp:

```python
# Experiment 1: Baseline
lstm_units=64, dropout=0.3, lr=1e-3

# Experiment 2: Deeper
lstm_units=128, dropout=0.4, lr=1e-3

# Experiment 3: Lower LR
lstm_units=64, dropout=0.3, lr=5e-4

# Experiment 4: Larger batch
lstm_units=64, dropout=0.3, batch_size=512
```

Ghi lại kết quả và so sánh.

#### **Learning Rate Schedule**

Nếu loss không giảm:
- Giảm learning rate: `1e-3 → 5e-4 → 1e-4`
- Hoặc dùng ReduceLROnPlateau (đã tích hợp sẵn)

---

## 🔧 Troubleshooting

### ❌ Lỗi: "FileNotFoundError: data/raw/bot_iot.csv"

**Giải pháp**:
```bash
# Kiểm tra đường dẫn
ls data/raw/

# Nếu file không tồn tại, download Bot-IoT dataset
# Sau đó đặt vào data/raw/
```

### ❌ Lỗi: "KeyError: 'attack'"

**Nguyên nhân**: Tên cột nhãn không đúng

**Giải pháp**:
```python
# Kiểm tra tên cột trong dataset
import pandas as pd
df = pd.read_csv('data/raw/bot_iot.csv')
print(df.columns)

# Chỉnh trong src/config.py
label_column = "tên_cột_nhãn_đúng"
```

### ❌ Model không học (loss không giảm)

**Kiểm tra**:
1. Data có bị lỗi không? (NaN, inf)
2. Features đã được normalize chưa? (StandardScaler)
3. Learning rate có quá cao không?

**Giải pháp**:
```python
# Giảm learning rate
learning_rate = 1e-4  # thay vì 1e-3

# Hoặc thử optimizer khác
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9)
```

### ❌ Out of Memory (OOM)

**Giải pháp**:
```python
# Giảm batch size
batch_size = 128  # hoặc 64

# Hoặc giảm model size
lstm_units = 32
dense_units = 16
```

### ❌ Training quá chậm

**Tăng tốc**:
1. Giảm số epochs
2. Tăng batch size
3. Sử dụng GPU (nếu có)
4. Giảm kích thước model (dùng lightweight config)

---

## 📝 Best Practices

### 1. Quy trình thử nghiệm

```
1. Quick test với lightweight config (kiểm tra pipeline)
2. Baseline với default config
3. Thử nghiệm các config khác (deep, sequence)
4. Hyperparameter tuning
5. Chọn model tốt nhất dựa trên validation metrics
6. Đánh giá trên test set
```

### 2. Logging và Tracking

- **Ghi lại mọi experiment**: config, metrics, training time
- **Sử dụng TensorBoard**: monitor real-time
- **Version control**: commit code sau mỗi experiment thành công

### 3. Reproducibility

```python
# Set random seed
random_state = 42

# Trong config.py
random_state: int = 42

# Trong numpy, tensorflow
np.random.seed(42)
tf.random.set_seed(42)
```

---

## 🎯 Kết luận và Khuyến nghị

### Quy trình đề xuất:

1. **Bắt đầu**: `default` config với `class_weight=True`
2. **Nếu recall thấp**: Thử `SMOTE=True` hoặc điều chỉnh threshold
3. **Nếu muốn tốc độ**: Dùng `lightweight` config
4. **Nếu muốn performance cao**: Dùng `deep` config
5. **Nếu có temporal data**: Thử `sequence` config

### Metrics mục tiêu:

- **Accuracy**: ≥ 95%
- **Precision**: ≥ 90%
- **Recall**: ≥ 95% (quan trọng nhất)
- **F1-Score**: ≥ 92%
- **ROC AUC**: ≥ 0.95

### Next Steps:

- Thử nghiệm các kiến trúc khác (stacked LSTM, attention mechanism)
- Feature engineering (chọn features quan trọng)
- Ensemble methods (kết hợp nhiều models)
- Deploy model thành API/service

---

## 📚 Tài liệu tham khảo

- Bot-IoT Dataset: https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/
- LSTM: https://colah.github.io/posts/2015-08-Understanding-LSTMs/
- TensorFlow/Keras: https://www.tensorflow.org/guide/keras
- Imbalanced Learning: https://imbalanced-learn.org/

---

**Good luck với training! 🚀**
