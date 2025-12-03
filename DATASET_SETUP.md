# Hướng dẫn Setup Dataset cho IoT DDoS Detection

## 📋 Tổng quan

Dự án này hỗ trợ nhiều dataset IoT DDoS khác nhau. Hướng dẫn này sẽ giúp bạn download và setup dataset phù hợp.

---

## 🎯 Dataset Khuyến nghị: Bot-IoT

**Đặc điểm:**
- **Kích thước**: ~16.7GB (CSV), ~69.3GB (PCAP)
- **Số mẫu**: 72+ triệu records
- **Labels**: Normal, DDoS, DoS, Reconnaissance, Theft
- **Năm**: 2018, UNSW Canberra

---

## 🚀 Cách 1: IMPACT CyberTrust (Khuyến nghị - Dễ nhất) ⭐

**Link**: [https://www.impactcybertrust.org/dataset_view?idDataset=1296](https://www.impactcybertrust.org/dataset_view?idDataset=1296)

**Ưu điểm**:
- Miễn phí, dễ download
- Có cả CSV và PCAP formats
- Không cần setup phức tạp

**Hướng dẫn**:
1. Truy cập link trên
2. Click "Download" (có thể cần đăng ký miễn phí)
3. Chọn CSV format (~16.7GB)
4. Giải nén vào `data/raw/`

---

## 🔄 Cách 2: OpenML (Dễ truy cập)

**Link**: [https://www.openml.org/d/42072](https://www.openml.org/d/42072)

**Ưu điểm**:
- Dễ dàng download
- Format CSV sẵn sàng sử dụng

---

## 📦 Cách 3: Kaggle Alternatives

### **CIC-BoT-IoT** (Với CICFlowmeter features)

**Link**: [https://www.kaggle.com/datasets/dhoogla/cicbotiot](https://www.kaggle.com/datasets/dhoogla/cicbotiot)

```bash
# Setup Kaggle CLI
pip install kaggle

# Setup API token (xem bước dưới)

# Download
kaggle datasets download -d dhoogla/cicbotiot -p data/raw/ --unzip
```

### **NF-BoT-IoT** (NetFlow version)

**Link**: [https://www.kaggle.com/datasets/dhoogla/nfbotiot](https://www.kaggle.com/datasets/dhoogla/nfbotiot)

```bash
kaggle datasets download -d dhoogla/nfbotiot -p data/raw/ --unzip
```

### Setup Kaggle API:

1. Đăng nhập [Kaggle](https://www.kaggle.com)
2. Vào [Settings](https://www.kaggle.com/settings) → API → Create New Token
3. Di chuyển `kaggle.json`:
   ```bash
   # Linux/Mac
   mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

   # Windows
   mkdir %USERPROFILE%\.kaggle && move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\
   ```

---

## 🆕 Cách 4: Dataset Mới Hơn (Thay thế)

### **CIC IoT-DIAD 2024** (Mới nhất)

- **Link**: [https://www.unb.ca/cic/datasets/iot-diad-2024.html](https://www.unb.ca/cic/datasets/iot-diad-2024.html)
- 33 attacks, 105 devices
- Cần chỉnh `label_column` trong `src/config.py`

### **CICIoT2023**

- **Link**: [https://www.unb.ca/cic/datasets/iotdataset-2023.html](https://www.unb.ca/cic/datasets/iotdataset-2023.html)

### **IoT-DH Dataset**

- **Link**: [https://data.mendeley.com/datasets/8dns3xbckv/1](https://data.mendeley.com/datasets/8dns3xbckv/1)
- ~2GB, dễ download

---

## ⚙️ Sau khi Download

```bash
# Kiểm tra file
ls -lh data/raw/

# Đổi tên (nếu cần)
mv data/raw/UNSW_2018_IoT_Botnet_Dataset_*.csv data/raw/bot_iot.csv

# Hoặc dùng trực tiếp:
python src/train_lstm.py --config default --data data/raw/your_file.csv

# Test load
python -c "import pandas as pd; df=pd.read_csv('data/raw/bot_iot.csv'); print(f'Shape: {df.shape}')"
```

---

## 🔍 Troubleshooting

### File quá lớn?

Dùng subset:
```python
import pandas as pd
df = pd.read_csv('data/raw/bot_iot.csv', nrows=100000)  # 100K rows
df.to_csv('data/raw/bot_iot_sample.csv', index=False)
```

### Label column khác?

Kiểm tra và cập nhật config:
```python
# Kiểm tra
df = pd.read_csv('data/raw/bot_iot.csv')
print(df.columns)

# Cập nhật src/config.py
label_column = "tên_cột_đúng"
```

---

## ✅ Checklist

- [ ] Dataset đã download
- [ ] File CSV có thể đọc
- [ ] Test load thành công
- [ ] Chỉnh `label_column` (nếu cần)

---

## 🎯 Khuyến nghị

1. **Bắt đầu**: IMPACT CyberTrust (dễ nhất)
2. **Backup**: Kaggle CIC-BoT-IoT
3. **Test ngay**: `python src/train_lstm.py --config lightweight --data data/raw/bot_iot.csv`

**Good luck! 🚀**
