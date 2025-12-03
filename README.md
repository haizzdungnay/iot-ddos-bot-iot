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

Cấu trúc thư mục được sử dụng:

```text
iot-ddos-bot-iot/
├─ .venv/                     # virtual environment (Python)
├─ data/
│  ├─ raw/                    # Bot-IoT gốc (CSV)
│  └─ processed/              # dữ liệu sau tiền xử lý (CSV đã làm sạch, chuẩn hóa)
├─ models/                    # lưu model LSTM đã huấn luyện
├─ notebooks/                 # Jupyter notebooks (nếu dùng để thử nghiệm)
├─ src/
│  ├─ __init__.py
│  ├─ preprocess.py           # hàm load + tiền xử lý dữ liệu Bot-IoT
│  └─ train_lstm.py           # huấn luyện & đánh giá mô hình LSTM
├─ requirements.txt           # danh sách thư viện Python
├─ README.md                  # mô tả đề tài (file này)
└─ .gitignore                 # nếu dùng git
