# Hướng dẫn Setup Dataset cho IoT DDoS Detection

## 📋 Tổng quan

Dự án này hỗ trợ nhiều dataset IoT DDoS khác nhau. Hướng dẫn này sẽ giúp bạn download và setup dataset phù hợp.

---

## 🎯 Dataset Khuyến nghị

### **Bot-IoT Dataset** (Khuyến nghị)

Bot-IoT là dataset được thiết kế đặc biệt cho IoT botnet attacks, bao gồm DDoS, DoS, và các loại tấn công khác.

**Đặc điểm:**
- **Kích thước**: Full (~16GB), 5% sample (~800MB)
- **Số mẫu**: Hàng triệu flows
- **Labels**: Normal, DDoS, DoS, Reconnaissance, Theft
- **Format**: CSV với nhiều features network

---

## 🚀 Cách 1: Download từ Kaggle (Dễ nhất)

### Bước 1: Tạo tài khoản Kaggle

1. Truy cập [https://www.kaggle.com](https://www.kaggle.com)
2. Đăng ký tài khoản miễn phí (hoặc đăng nhập nếu đã có)

### Bước 2: Setup Kaggle API

#### **Option A: Kaggle CLI (Khuyến nghị)**

1. **Cài đặt Kaggle CLI:**
   ```bash
   pip install kaggle
   ```

2. **Lấy API Token:**
   - Đăng nhập Kaggle
   - Vào [https://www.kaggle.com/settings](https://www.kaggle.com/settings)
   - Scroll xuống phần "API"
   - Click "Create New Token"
   - File `kaggle.json` sẽ được download

3. **Setup API Token:**

   **Linux/Mac:**
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

   **Windows:**
   ```powershell
   # Tạo thư mục
   mkdir C:\Users\<YourUsername>\.kaggle

   # Di chuyển file kaggle.json vào đó
   move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\kaggle.json
   ```

4. **Download Dataset:**

   **Option A: Full Dataset (~16GB)**
   ```bash
   kaggle datasets download -d vigneshvenkateswaran/bot-iot -p data/raw/ --unzip
   ```

   **Option B: 5% Sample (~800MB) - Khuyến nghị cho bắt đầu**
   ```bash
   kaggle datasets download -d vigneshvenkateswaran/bot-iot-5-data -p data/raw/ --unzip
   ```

   **Option C: All Features 5% Sample**
   ```bash
   kaggle datasets download -d majedjaber/bot-iot-all-features-5-sample -p data/raw/ --unzip
   ```

#### **Option B: Download thủ công từ Web**

1. Truy cập một trong các link:
   - Full: [https://www.kaggle.com/datasets/vigneshvenkateswaran/bot-iot](https://www.kaggle.com/datasets/vigneshvenkateswaran/bot-iot)
   - 5% Sample: [https://www.kaggle.com/datasets/vigneshvenkateswaran/bot-iot-5-data](https://www.kaggle.com/datasets/vigneshvenkateswaran/bot-iot-5-data)

2. Click nút "Download" (màu xanh)

3. Chờ download hoàn tất

4. Giải nén file:
   ```bash
   # Linux/Mac
   unzip bot-iot.zip -d data/raw/

   # Windows: Chuột phải > Extract All > Chọn data/raw/
   ```

### Bước 3: Kiểm tra và đổi tên file

```bash
# Liệt kê các file đã download
ls -lh data/raw/

# Dataset thường có tên dạng: UNSW_2018_IoT_Botnet_Dataset_*.csv
# Ví dụ: UNSW_2018_IoT_Botnet_Dataset_5.csv

# Option 1: Đổi tên thành bot_iot.csv
mv data/raw/UNSW_2018_IoT_Botnet_Dataset_*.csv data/raw/bot_iot.csv

# Option 2: Giữ nguyên tên và dùng --data flag khi train
python src/train_lstm.py --config default --data data/raw/UNSW_2018_IoT_Botnet_Dataset_5.csv
```

### Bước 4: Verify dataset

```python
import pandas as pd

# Load để kiểm tra
df = pd.read_csv('data/raw/bot_iot.csv')
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nLabel distribution:")
print(df['attack'].value_counts())  # hoặc 'category' tùy dataset
```

---

## 🔄 Cách 2: Dataset thay thế

Nếu không download được Bot-IoT, có thể dùng các dataset sau:

### **1. CIC IoT-DIAD 2024** (Dataset mới nhất)

**Đặc điểm:**
- Dataset IoT mới nhất (2024)
- 33 loại attacks
- 105 IoT devices
- Format: CSV với flow/packet features

**Download:**
```bash
# Truy cập và download từ:
# https://www.unb.ca/cic/datasets/iot-diad-2024.html

# Sau khi download, giải nén:
unzip IoT-DIAD-2024.zip -d data/raw/
```

**Lưu ý:** Cần chỉnh `label_column` trong `src/config.py`:
```python
# src/config.py
label_column = "Label"  # Hoặc tên cột tương ứng
```

### **2. CICIoT2023**

**Download:**
- Link: [https://www.unb.ca/cic/datasets/iotdataset-2023.html](https://www.unb.ca/cic/datasets/iotdataset-2023.html)
- Tương tự CIC IoT-DIAD 2024

### **3. IoT-DH Dataset**

**Download:**
```bash
# Truy cập Mendeley Data:
# https://data.mendeley.com/datasets/8dns3xbckv/1

# Click "Download" (miễn phí, cần đăng ký Mendeley)
```

---

## ⚙️ Cấu hình cho Dataset khác Bot-IoT

Nếu dùng dataset khác, cần chỉnh config:

### 1. Kiểm tra tên cột nhãn

```python
import pandas as pd
df = pd.read_csv('data/raw/your_dataset.csv')
print(df.columns)
```

### 2. Cập nhật config

Mở `src/config.py` và chỉnh:

```python
@dataclass
class DataConfig:
    # ...
    # Đổi tên cột nhãn cho đúng
    label_column: str = "Label"  # Thay "attack" → "Label" nếu dataset dùng "Label"
```

### 3. Kiểm tra encoding

Dataset khác nhau có thể có label encoding khác:

**Bot-IoT:**
- Labels: "Normal", "DDoS", "DoS", "Reconnaissance", "Theft"

**CIC IoT-DIAD 2024:**
- Labels: "Benign", "DDoS", "DoS", "Recon", v.v.

**Giải pháp:** Code đã tự động xử lý, chỉ cần đảm bảo có class "Normal" hoặc "Benign":

Nếu dataset dùng "Benign" thay vì "Normal", chỉnh trong `src/preprocess.py`:

```python
# Line 80-82 trong preprocess.py
df['label_binary'] = df[label_col].apply(
    lambda x: 0 if str(x).lower() in ['normal', 'benign'] else 1
)
```

---

## 📊 So sánh các Dataset

| Dataset | Kích thước | Năm | Devices | Attack Types | Dễ download | Khuyến nghị |
|---------|-----------|------|---------|--------------|-------------|------------|
| **Bot-IoT (Kaggle)** | 800MB-16GB | 2018 | IoT Botnet | 5 types | ⭐⭐⭐⭐⭐ | ✅ Bắt đầu |
| **CIC IoT-DIAD 2024** | ~10GB | 2024 | 105 devices | 33 attacks | ⭐⭐⭐ | ✅ Dataset mới |
| **CICIoT2023** | ~5GB | 2023 | Multiple | 33 attacks | ⭐⭐⭐ | ✅ Thay thế tốt |
| **IoT-DH** | ~2GB | 2024 | IoT/OT | DDoS focus | ⭐⭐⭐⭐ | ✅ Nhẹ hơn |

---

## 🔍 Troubleshooting

### ❌ Lỗi: "kaggle: command not found"

**Giải pháp:**
```bash
# Cài lại Kaggle CLI
pip install --upgrade kaggle

# Kiểm tra
kaggle --version
```

### ❌ Lỗi: "401 - Unauthorized"

**Nguyên nhân:** API token chưa setup đúng

**Giải pháp:**
```bash
# Kiểm tra file kaggle.json tồn tại
ls ~/.kaggle/kaggle.json

# Kiểm tra quyền (Linux/Mac)
chmod 600 ~/.kaggle/kaggle.json

# Kiểm tra nội dung
cat ~/.kaggle/kaggle.json
# Phải có dạng: {"username":"...","key":"..."}
```

### ❌ Lỗi: "403 - Forbidden"

**Giải pháp:**
1. Đăng nhập Kaggle trên web
2. Vào trang dataset
3. Click "Download" một lần (để chấp nhận terms)
4. Sau đó dùng CLI sẽ work

### ❌ File CSV quá lớn, không load được

**Giải pháp:**
```python
# Load từng phần (chunking)
import pandas as pd

chunks = []
for chunk in pd.read_csv('data/raw/bot_iot.csv', chunksize=100000):
    chunks.append(chunk)
    if len(chunks) >= 10:  # Lấy 1M rows đầu
        break

df = pd.concat(chunks, ignore_index=True)
```

Hoặc dùng **5% sample** thay vì full dataset.

### ❌ Lỗi: "KeyError: 'attack'"

**Nguyên nhân:** Tên cột nhãn khác

**Giải pháp:**
```python
# Kiểm tra tên cột
df = pd.read_csv('data/raw/bot_iot.csv')
print(df.columns)

# Cập nhật trong src/config.py
label_column = "tên_cột_đúng"
```

---

## ✅ Checklist sau khi Setup

- [ ] Dataset đã download về `data/raw/`
- [ ] File CSV có thể đọc được
- [ ] Đã kiểm tra tên cột nhãn
- [ ] Đã cập nhật `label_column` trong config (nếu cần)
- [ ] Test load data:
  ```bash
  python -c "import pandas as pd; df=pd.read_csv('data/raw/bot_iot.csv'); print(df.shape)"
  ```

---

## 🎯 Khuyến nghị cho người mới

1. **Bắt đầu với Bot-IoT 5% sample từ Kaggle**
   - Nhẹ (~800MB)
   - Dễ download
   - Đủ để test pipeline

2. **Test pipeline trước:**
   ```bash
   python src/train_lstm.py --config lightweight --data data/raw/bot_iot.csv
   ```

3. **Sau khi pipeline chạy OK, chuyển sang full dataset** (nếu cần performance tốt hơn)

---

## 📞 Support

Nếu gặp vấn đề khi download dataset:

1. Kiểm tra [TRAINING_GUIDE.md](TRAINING_GUIDE.md#troubleshooting)
2. Tạo issue trên GitHub repository
3. Tham khảo documentation của dataset:
   - Bot-IoT Kaggle: Comments section trong trang dataset
   - CIC: https://www.unb.ca/cic/datasets/

---

**Good luck! 🚀**
