# Hướng dẫn Setup Dataset cho IoT DDoS Detection

## 📋 Tổng quan

Dự án này hỗ trợ nhiều dataset IoT DDoS. Hướng dẫn này giúp bạn download và setup dataset.

---

## 🎯 Dataset: Bot-IoT

**Đặc điểm:**
- **Kích thước**: ~16.7GB (CSV), ~69.3GB (PCAP)
- **Số mẫu**: 72+ triệu records
- **Labels**: Normal, DDoS, DoS, Reconnaissance, Theft
- **Năm**: 2018, UNSW Canberra

---

## 🚀 Cách 1: UNSW OneDrive (Khuyến nghị - Dễ nhất) ⭐

### Link chính thức:
**[UNSW Bot-IoT OneDrive](https://unsw-my.sharepoint.com/personal/z5131399_ad_unsw_edu_au/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fz5131399%5Fad%5Funsw%5Fedu%5Fau%2FDocuments%2FBot%2DIoT%5FDataset&ga=1)**

### Hướng dẫn download:

1. **Truy cập link OneDrive** ở trên

2. **Chọn file để download**:
   - **CSV format** (~16.7 GB) - Khuyến nghị cho ML
   - **PCAP format** (~69.3 GB) - Nếu cần raw packets

3. **Click "Download"** trên OneDrive interface

4. **Chờ download hoàn tất**

5. **Giải nén và setup**:
   ```bash
   # Giải nén vào data/raw/
   unzip bot-iot.zip -d data/raw/

   # Hoặc nếu file là tar.gz:
   tar -xzvf bot-iot.tar.gz -C data/raw/

   # Kiểm tra file
   ls -lh data/raw/

   # Đổi tên (nếu cần)
   mv data/raw/UNSW_2018_IoT_Botnet_Dataset_*.csv data/raw/bot_iot.csv

   # Hoặc dùng trực tiếp:
   python src/train_lstm.py --config default --data data/raw/UNSW_2018_IoT_Botnet_Dataset_5.csv
   ```

---

## 🔄 Cách 2: Nguồn Thay Thế

Nếu link OneDrive không hoạt động, thử:

### **IMPACT CyberTrust** (Mirror miễn phí)
- **Link**: [https://www.impactcybertrust.org/dataset_view?idDataset=1296](https://www.impactcybertrust.org/dataset_view?idDataset=1296)
- Miễn phí, dễ download

### **OpenML**
- **Link**: [https://www.openml.org/d/42072](https://www.openml.org/d/42072)
- Format CSV sẵn sàng

### **UNSW Research**
- **Link**: [https://research.unsw.edu.au/projects/bot-iot-dataset](https://research.unsw.edu.au/projects/bot-iot-dataset)
- Trang chính thức

---

## 📦 Cách 3: Kaggle Alternatives

### **CIC-BoT-IoT** (CICFlowmeter features)
**Link**: [https://www.kaggle.com/datasets/dhoogla/cicbotiot](https://www.kaggle.com/datasets/dhoogla/cicbotiot)

```bash
# Setup Kaggle CLI
pip install kaggle

# Setup API token (xem bên dưới)

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
# Test load dữ liệu
python -c "import pandas as pd; df=pd.read_csv('data/raw/bot_iot.csv'); print(f'Shape: {df.shape}')"

# Kiểm tra columns
python -c "import pandas as pd; df=pd.read_csv('data/raw/bot_iot.csv'); print(df.columns.tolist())"

# Kiểm tra labels
python -c "import pandas as pd; df=pd.read_csv('data/raw/bot_iot.csv'); print(df['attack'].value_counts())"
```

---

## 🔍 Troubleshooting

### File quá lớn?
```python
# Lấy subset 100K rows
import pandas as pd
df = pd.read_csv('data/raw/bot_iot.csv', nrows=100000)
df.to_csv('data/raw/bot_iot_sample.csv', index=False)
```

### Label column khác?
```python
# Kiểm tra
df = pd.read_csv('data/raw/bot_iot.csv')
print(df.columns)

# Cập nhật src/config.py
label_column = "tên_cột_đúng"
```

### Link OneDrive không hoạt động?
- Thử IMPACT CyberTrust
- Hoặc Kaggle CIC-BoT-IoT
- Hoặc OpenML

---

## ✅ Checklist

- [ ] Dataset đã download
- [ ] File CSV có thể đọc
- [ ] Test load thành công
- [ ] Chỉnh `label_column` (nếu cần)

---

## 🎯 Khuyến nghị

1. **Bắt đầu**: UNSW OneDrive (dễ nhất, chính thức)
2. **Backup**: IMPACT CyberTrust hoặc OpenML
3. **Alternative**: Kaggle CIC-BoT-IoT
4. **Test ngay**: `python src/train_lstm.py --config lightweight --data data/raw/bot_iot.csv`

**Good luck! 🚀**
