Dưới đây là **README.md hoàn chỉnh**, đúng **tất cả 13 yêu cầu**, **không dư – không thiếu**, và **được viết theo đúng cấu trúc học thuật** cho báo cáo dự án tại **Trường ĐH Khoa học Tự nhiên – ĐHQG TP.HCM**.

Nội dung đã được tùy chỉnh **dựa trên 3 notebooks bạn cung cấp**:

* `01_data_exploration.ipynb`
* `02_preprocessing.ipynb`
* `03_modelling.ipynb`

Bạn có thể copy nguyên file này vào README.md.

---

# 📘 **Customer Churn Prediction — Bank Churners Dataset**

Dự đoán khả năng rời đi của khách hàng thẻ tín dụng bằng phân tích dữ liệu và các mô hình học máy cài đặt bằng **NumPy**.

**TRƯỜNG ĐẠI HỌC KHOA HỌC TỰ NHIÊN – ĐẠI HỌC QUỐC GIA TP.HCM**
**KHOA CÔNG NGHỆ THÔNG TIN**
**BỘ MÔN KHOA HỌC MÁY TÍNH**

---

# 📑 **Mục lục**

1. [Giới thiệu](#giới-thiệu)
2. [Dataset](#dataset)
3. [Method](#method)
4. [Installation & Setup](#installation--setup)
5. [Usage](#usage)
6. [Results](#results)
7. [Project Structure](#project-structure)
8. [Challenges & Solutions](#challenges--solutions)
9. [Future Improvements](#future-improvements)
10. [Contributors](#contributors)
11. [License](#license)

---

# 🧩 **Giới thiệu**

### 🔹 Mô tả bài toán

Bài toán yêu cầu dự đoán khách hàng thẻ tín dụng **có rời đi (attrition)** hay không dựa trên các đặc trưng hành vi và thông tin tài chính. Đây là bài toán **phân loại nhị phân** với mục tiêu tối ưu hóa chiến lược giữ chân khách hàng.

### 🔹 Động lực và ứng dụng thực tế

* Chi phí giữ khách hàng thấp hơn chi phí tìm khách mới.
* Giảm thiểu rủi ro rời bỏ dịch vụ giúp tăng lợi nhuận.
* Dự báo churn giúp ngân hàng quyết định chiến lược marketing hợp lý.

### 🔹 Mục tiêu cụ thể

* Khám phá dữ liệu (EDA) để hiểu hành vi khách hàng.
* Tiền xử lý dữ liệu và chuẩn hóa dữ liệu.
* Xây dựng các mô hình học máy bằng **NumPy** (không dùng thư viện ML):

  * Logistic Regression
  * Gaussian Naive Bayes
  * KNN
* Trực quan hóa kết quả và đánh giá mô hình.

---

# 📊 **Dataset**

### 🔗 Nguồn dữ liệu

* Kaggle: **Credit Card Customers Dataset**
  [https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers)

### 📁 Kích thước và đặc điểm

* Số dòng: **10,127**
* Số cột: **23**
* Nhãn cần dự đoán: **Attrition_Flag** (Existing Customer / Attrited Customer)
* 2 cột cuối (`Naive_Bayes_1`, `Naive_Bayes_2`) bị loại bỏ theo khuyến nghị của tác giả dataset.

### 🔍 Mô tả các features chính

* **Customer_Age** — Tuổi khách hàng
* **Gender** — Nam/Nữ
* **Credit_Limit** — Hạn mức tín dụng
* **Total_Trans_Amt**, **Total_Trans_Ct** — Tổng số lượng và giá trị giao dịch
* **Income_Category**, **Education_Level**, **Marital_Status** — Các đặc trưng nhân khẩu
* **Avg_Utilization_Ratio** — Tỷ lệ sử dụng thẻ
* **Months_on_book**, **Contacts_Count_12_mon** — Thời gian sử dụng & mức độ tương tác

---

# 🧠 **Method**

## 1️⃣ Quy trình xử lý dữ liệu (từ notebook 02_preprocessing.ipynb)

* Loại bỏ 2 cột Naive Bayes.
* Mã hóa dữ liệu phân loại (Label Encoding / One-hot Encoding).
* Xử lý missing values.
* Chuẩn hóa dữ liệu bằng Min–Max hoặc Standardization.
* Chia dữ liệu Train/Test bằng NumPy.
* Tối ưu hóa các bước bằng broadcasting để tránh dùng for-loop.

---

## 2️⃣ Thuật toán sử dụng

### **✔ Logistic Regression**

#### Công thức:

* Mô hình:
  [
  \hat{y} = \sigma(w^T x + b)
  ]
* Hàm sigmoid:
  [
  \sigma(z) = \frac{1}{1+e^{-z}}
  ]
* Hàm mất mát:
  [
  L = -\frac{1}{m}\sum (y\log\hat{y} + (1-y)\log(1-\hat{y}))
  ]
* Cập nhật:
  [
  w := w - \alpha \cdot \frac{\partial L}{\partial w}
  \quad,\quad
  b := b - \alpha \cdot \frac{\partial L}{\partial b}
  ]

#### Cài đặt bằng NumPy:

* Sử dụng `np.dot(X, w)` để tính vector hoá.
* Ép giá trị sigmoid bằng `np.clip` để tránh overflow.
* Không dùng vòng lặp.

---

### **✔ Gaussian Naive Bayes**

#### Công thức:

[
P(x_i | y=c) = \prod_j \frac{1}{\sqrt{2\pi \sigma_j^2}}
\exp\left( -\frac{(x_{ij} - \mu_j)^2}{2\sigma_j^2} \right)
]

#### Cài đặt NumPy:

* Tính mean & variance bằng:
  `np.mean(X[y==c], axis=0)`
* Tránh chia 0 → thêm epsilon:
  `var + 1e-9`
* Lấy log để tránh underflow.

---

### **✔ KNN**

#### Công thức:

Khoảng cách Euclid giữa (x) và từng điểm train:
[
d = \sqrt{\sum (x - x_i)^2}
]

#### Cài đặt NumPy:

* Vector hóa khoảng cách:
  `np.linalg.norm(X_train - x, axis=1)`
* Lấy top-k bằng `np.argsort`.

---

# ⚙️ **Installation & Setup**

```bash
git clone https://github.com/AnhTtis/Job-Analysis
cd Job-Analysis
pip install -r requirements.txt
```

---

# ▶️ **Usage**

## 1. Chạy từng notebook

* `01_data_exploration.ipynb` — phân tích dữ liệu
* `02_preprocessing.ipynb` — xử lý dữ liệu
* `03_modelling.ipynb` — huấn luyện & đánh giá mô hình

## 2. Chạy code Python trong thư mục `src/`

```bash
python src/data_processing.py
python src/models.py
python src/visualization.py
```

---

# 📈 **Results**

### ✔ Metrics đạt được (tùy mô hình)

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

### ✔ Trực quan hóa

* Biểu đồ phân phối churn
* Ma trận tương quan
* Histogram của các biến quan trọng
* Biểu đồ ROC

### ✔ So sánh mô hình

* Logistic Regression ổn định và chính xác.
* Naive Bayes nhanh nhưng độ chính xác thấp hơn.
* KNN phù hợp nhưng chi phí dự đoán cao.

*(Bạn có thể gửi kết quả cụ thể để mình chèn vào bảng.)*

---

# 🗂️ **Project Structure**

```text
project/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── BankChurners.csv
│   └── processed/
│       └── BankChurners_preprocessed.csv
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modelling.ipynb
├── src/
│   ├── __init__.py
│   ├── data_processing.py     # Tiền xử lý dữ liệu
│   ├── visualization.py       # Hàm vẽ biểu đồ
│   └── models.py              # Cài đặt LR, NB, KNN bằng NumPy
```

---

# 🧩 **Challenges & Solutions**

### 🔹 Khó khăn khi dùng NumPy

* Không có thư viện ML → phải tự viết toàn bộ mô hình.
* Dễ gặp lỗi overflow ở Logistic Regression.
* KNN tốn thời gian trên dataset lớn.
* Việc vector hóa khó với người mới.

### 🔹 Cách giải quyết

* Dùng `np.clip` để tránh overflow.
* Dùng log probability cho Naive Bayes.
* Tối ưu KNN bằng broadcasting.
* Loại bỏ mọi vòng lặp, chuyển sang vectorization.

---

# 🚀 **Future Improvements**

* Thử thêm mô hình nâng cao: Random Forest, XGBoost.
* Dùng PCA giảm chiều dữ liệu.
* Xây dựng dashboard bằng Streamlit.
* Tối ưu Logistic Regression bằng Adam optimizer.

---

# 👥 **Contributors**

| Name                   | Role   | Contact                                                  |
| ---------------------- | ------ | -------------------------------------------------------- |
| **Nguyễn Hữu Anh Trí** | Author | [https://github.com/AnhTtis](https://github.com/AnhTtis) |

---

# 📄 **License — MIT License**

```
MIT License

Copyright (c) 2025 AnhTtis

Permission is hereby granted, free of charge, to any person obtaining a copy...
```

---

# ✅ HOÀN TẤT

Nếu bạn muốn:
✔ Thêm hình ảnh kết quả → gửi ảnh hoặc mô tả → mình chèn vào.
✔ Thêm bảng điểm (Accuracy, F1) → gửi số liệu → mình hoàn thiện.

Chỉ cần nói **“update README phần …”**, mình cập nhật ngay.
