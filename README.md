# **Customer Churn Prediction — Bank Churners Dataset**

Dự đoán khả năng rời đi của khách hàng thẻ tín dụng bằng phân tích dữ liệu và các mô hình học máy cài đặt bằng **NumPy**.

---

# **Mục lục**

- [I. Giới thiệu](#giới-thiệu)
  - [I.1. Mô tả bài toán](#i1--mô-tả-bài-toán)
  - [I.2 Động lực và ứng dụng thực tế](#ii2-động-lực-và-ứng-dụng-thực-tế)
  - [I.3 Mục tiêu cụ thể](#ii3-mục-tiêu-cụ-thể)
- [Dataset](#dataset)
  - [II.1 Nguồn dữ liệu](#ii1-nguồn-dữ-liệu)
  - [II.2 Kích thước và đặc điểm](#ii2-kích-thước-và-đặc-điểm)
  - [II. 3 Mô tả các feature](#ii3-mô-tả-các-features)
- [Method](#method)
4. [Installation & Setup](#installation--setup)
5. [Usage](#usage)
6. [Results](#results)
7. [Project Structure](#project-structure)
8. [Challenges & Solutions](#challenges--solutions)
9. [Future Improvements](#future-improvements)
10. [Contributors](#contributors)
11. [License](#license)

---

# I. **Giới thiệu**

## I.1  Mô tả bài toán

Bài toán yêu cầu dự đoán khách hàng thẻ tín dụng **có rời đi** (`attrition`) hay không dựa trên các đặc trưng hành vi và thông tin tài chính. Đây là bài toán **phân loại nhị phân** (`binary classification`)  với mục tiêu tối ưu hóa chiến lược giữ chân khách hàng.

## II.2 Động lực và ứng dụng thực tế

* Khi phải đưa ra chính sách hoặc phương án trong việc giữ lại khách hành tiềm năng hay tìm kiếm khách hàng mới, ta nhận thấy chi phí giữ khách hàng thấp hơn chi phí tìm khách mới nhưng phải biết được những khách hàng nào còn ở lại.

* Có những chính sách giúp giữ lại nhiều khách hàng để giảm thiểu rủi ro rời bỏ dịch vụ, giúp tăng lợi nhuận.

* Dự báo được xu hướng của khách hàng giúp ngân hàng quyết định chiến lược marketing hợp lý.

## II.3 Mục tiêu cụ thể

* Khám phá dữ liệu (EDA) để hiểu hành vi khách hàng. Đồng thời, đưa ra một số khám phá thú vị (insights - nếu có) về dữ liệu

* Tiền xử lý dữ liệu và chuẩn hóa dữ liệu để chuẩn bị cho các mô hình học máy.

* Xây dựng các mô hình học máy bằng **NumPy** (không dùng thư viện ML):

  * Logistic Regression
  * Gaussian Naive Bayes
  * KNN

* Xây dựng một số hàm đánh giá mô hình

---

# II. **Dataset**

## II.1 Nguồn dữ liệu

* Kaggle: [Credit Card Customers Dataset](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers)

* Tên file: `BankChurners.csv` (Trong folder data)

* Giấy phép: [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)

## II.2 Kích thước và đặc điểm

* Số dòng: **10,127**
* Số cột: **23**
* Nhãn cần dự đoán: **Attrition_Flag** (Existing Customer / Attrited Customer)
* 2 cột cuối (`Naive_Bayes_1`, `Naive_Bayes_2`) bị loại bỏ theo khuyến nghị của tác giả dataset.

## II.3 Mô tả các features:

Bộ dữ liệu bao gồm **23 cột** như sau:

| Tên Cột | Mô Tả | 
| :--- | :--- | 
| `CLIENTNUM` | Mã định danh khách hàng (Unique ID) | 
| `Attrition_Flag` | Trạng thái hoạt động | 
| `Customer_Age` | Độ tuổi khách hàng |
| `Gender` | Giới tính | 
| `Dependent_count` | Số người phụ thuộc | 
| `Education_Level` | Trình độ học vấn | 
| `Marital_Status` | Tình trạng hôn nhân |
| `Income_Category` | Nhóm thu nhập hàng năm | 
| `Card_Category` | Loại thẻ tín dụng |
| `Months_on_book` | Thời gian gắn bó với ngân hàng |
| `Total_Relationship_Count` | Tổng số sản phẩm/dịch vụ sở hữu |
| `Months_Inactive_12_mon` | Số tháng không hoạt động (12 tháng qua) |
| `Contacts_Count_12_mon` | Số lần liên hệ ngân hàng (12 tháng qua) |
| `Credit_Limit` | Hạn mức tín dụng tối đa | 
| `Total_Revolving_Bal` | Tổng dư nợ quay vòng | 
| `Avg_Open_To_Buy` | Hạn mức khả dụng trung bình (mua sắm) | 
| `Total_Amt_Chng_Q4_Q1` | Tỷ lệ thay đổi số tiền giao dịch (Q4 vs Q1) |
| `Total_Trans_Amt` | Tổng tiền giao dịch (12 tháng qua) | 
| `Total_Trans_Ct` | Tổng số lần giao dịch (12 tháng qua) | 
| `Total_Ct_Chng_Q4_Q1` | Tỷ lệ thay đổi số lần giao dịch (Q4 vs Q1) | 
| `Avg_Utilization_Ratio` | Tỷ lệ sử dụng thẻ trung bình | 
| `Naive_Bayes_Classifier..._1` | Kết quả từ thuật toán Naive Bayes (Gốc) - Cột sinh ra từ quá trình xây dựng dữ liệu |
| `Naive_Bayes_Classifier..._2` | Kết quả từ thuật toán Naive Bayes (Gốc) - Cột sinh ra từ quá trình xây dựng dữ liệu |

---

# III. **Method**

## III.1 Quy trình Khám phá dữ liệu
* **File trình bày**: 01_data_exploration.ipynb trong folder notebooks
* **Lưu ý**: Trong quá trình thực hiện việc khám phá dữ liệu ta sẽ xây dựng một từ điển để chúng ta mapping dữ liệu từng cột thống qua tên cột
* **Quá trình thực hiện**: 
  - **Bước 1**: Xác định một số thông tin cơ bản của `Dataset`
    - Xác định số dòng và số cột của dữ liệu. 
    - Xác định tên của từng cột dữ liệu, có bao nhiêu cột định danh và bao nhiêu cột số
    - Xác định khoảng giá trị của từng cột: đối với cột định danh thì in ra các `các giá trị riêng biệt`, còn đối với các cột có giá trị gồm nhiều số thì in ra `số lớn nhất, số nhỏ nhất, số trung bình, trung vị và phương sai`.
    - Xác định kiểu dữ liệu của từng cột và kiểm tra số lượng dữ liệu bị thiếu

  - **Bước 2**: Biểu diễn phân phối của từng cột dữ liệu. Đối với các cột số thì dùng `biểu đồ tần số (Histogram)` kết hợp với `đường cong mật độ (KDE)` , còn đối với các cột định danh thì dùng `biểu đồ cột (Bar chart)` để nhận xét phân phối của từng cột (Ví dụ như: các cột số thì có thể có ` Phân phối chuẩn`, ` Phân phối lệch phải/trái`,v.v còn các cột định danh có thể nhận biết được độ lệch giữa các thành phần)

  - **Bước 3**: Sử dụng `biểu đồ tròn (Pie chart)` để hiển thị phần trăm các giá trị trong các cột định  danh

  - **Bước 4**: Sử dụng `biểu đồ hộp` để so sánh phân phối số liệu trong các cột giá trị số với cột đặc trưng `Attrition_Flag` (được chuẩn hóa thành 0 - `Attrited Customer` và 1 - `Existing Customer`)

  - **Bước 5:** Sử dụng `Countplot (Biểu đồ đếm tần suất được tình bày dưới dạng biểu đồ thanh)` để so sánh phân phối của các cột có giá trị định danh so với cột đặc trưng `Attrition_Flag`

  - **Bước 6:** Vẽ `ma trận tương quan (correlation matrix)` giữa các cột để ta có một số insights cần thiết cho quá trình xử lí dữ liệu và cây dựng model 
    - Đối với các cột số:
      1. Chuẩn hóa cột `Attrition_Flag` (`Existing Customer` &rarr; 1; `Attrited Customer` &rarr; 0)
      2. Sử dụng hệ số tương quan `Pearson` để tính toán và biểu diễn ma trận tương quan (`correlation matrix`) giữa các cột có giá trị số cũng như cột giá trị đặc trưng `Attrition_Flag`

    - Đối với các cột định danh: Sử dụng hệ số `Cramér's V` để tính toán và biểu diễn ma trận tương quan (`correlation matrix`) giữa các cột định danh

## III.2 Quy trình xử lý dữ liệu:
* **File trình bày**: 02_preprocessing.ipynb trong folder notebooks
* **Lưu ý**: Trước khi bắt đầu Xử lí dữ liệu, ta cũng sẽ xây dựng một từ điển để chúng ta mapping dữ liệu từng cột thông qua tên cột giống như trên
* **Quá trình thực hiện**:
  - **Bước 1**: Bỏ những cột không quan trọng trong việc thiết kế mô hình bao gồm   **`Avg_Open_To_Buy`** và  **`Gender`** (Rút ra được từ quá trình `Khám phá dữ liệu`)

  - **Bước 2**: Chuẩn hóa cột đặc trưng Attrition_Flag và các cột định danh. 

  - **Bước 3**: Sử dụng Z-Score Scaling cho các nhóm phân phối chuẩn

  - **Bước 4**:Tạo cột mới: `Avg_Total_Trans`bằng cách lấy thương của phép chia `Total_Trans_Amt` và `Total_Trans_Ct`

  - **Bước 5:** Sữ dụng Log Scaling cho những cột có phân phối lệch phải, có giá trị lớn 

  - **Bước 6:**  Sử dụng MinMax cho những cột đã nằm trong phạm vi [0, 1]

  - **Bước 7:** Lưu lại dưới liệu dưới dạng file mới `BankChurners_preprocessed.csv` trong file data


---

## III.3 Quy trình xây dựng model:
### III.3.1 Thuật toán Logistic Regression

**Định nghĩa:**
Đây là thuật toán học máy có giám sát (Supervised Learning) chuyên dụng cho bài toán phân loại nhị phân (Binary Classification), đưa ra dự đoán dưới dạng xác suất (0 hoặc 1).

**Quy trình xây dựng và tối ưu mô hình:**

Quá trình huấn luyện được thực hiện qua các bước lặp (epochs) với cơ chế Gradient Descent:

**1. Khởi tạo tham số (Initialization)**
Thiết lập trạng thái ban đầu cho mô hình chưa được huấn luyện:
* Trọng số $w$ là vector 0 (`np.zeros`) và hệ số chệch $b = 0$.

**2. Quá trình Lan truyền xuôi (Forward Propagation)**
Tính toán dự đoán xác suất dựa trên dữ liệu đầu vào:
* **Tổ hợp tuyến tính:**
    $$z = w \cdot X + b$$
* **Hàm kích hoạt Sigmoid:**
    $$\hat{y} = \frac{1}{1 + e^{-z}}$$
* **Lưu ý kỹ thuật (NumPy):** Để đảm bảo tính ổn định số học và tránh lỗi tràn số (overflow) khi tính hàm mũ `np.exp`, giá trị của $z$ được giới hạn trong khoảng [-250, 250] bằng hàm `np.clip`.

**3. Tính toán Gradient (Backward Propagation)**
Tính đạo hàm của hàm mất mát để xác định hướng điều chỉnh tham số. Việc tính toán được **vector hóa** hoàn toàn bằng `np.dot` giúp tăng tốc độ xử lý so với vòng lặp thông thường:
* **Gradient của trọng số $w$:**
    $$dw = \frac{1}{m} X^T (\hat{y} - y)$$
* **Gradient của hệ số $b$:**
    $$db = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})$$

**4. Cập nhật tham số (Parameter Update)**
Điều chỉnh tham số ngược hướng Gradient để giảm thiểu sai số, với $\alpha$ là tốc độ học (learning rate):
* $w_{new} = w_{old} - \alpha \times dw$
* $b_{new} = b_{old} - \alpha \times db$

**5. Dự đoán (Prediction)**
Sau khi tối ưu hóa $w$ và $b$, mô hình đưa ra kết quả phân loại dựa trên ngưỡng xác suất (Threshold):
* Nếu $\hat{y} > 0.5 \Rightarrow$ Lớp 1.
* Ngược lại $\Rightarrow$ Lớp 0.

---

### III.3.2 Thuật toán K-Nearest Neighbors (KNN)

**Định nghĩa:**
Đây là thuật toán thuộc nhóm **Học lười (Lazy Learning)** và **Phi tham số (Non-parametric)** dùng cho bài toán phân loại. Khác với các mô hình học máy thông thường, KNN không huấn luyện để tìm ra bộ trọng số cố định mà trực tiếp ghi nhớ toàn bộ dữ liệu. Quyết định phân loại dựa trên sự tương đồng giữa dữ liệu mới và dữ liệu đã biết.

**Quy trình xây dựng mô hình:**

**1. Ghi nhớ dữ liệu (Training Phase)**
Mô hình chỉ thực hiện việc lưu trữ dữ liệu huấn luyện (`X_train`, `y_train`) vào bộ nhớ mà không thực hiện bất kỳ phép tính toán nào tại bước này.

**2. Dự đoán (Prediction Phase)**
Khi tiếp nhận dữ liệu đầu vào mới, thuật toán thực hiện chuỗi xử lý sau:

* **Tính toán khoảng cách (Vectorized Distance Calculation):**
    Mục tiêu là tính khoảng cách Euclidean giữa điểm dữ liệu mới và toàn bộ tập dữ liệu huấn luyện. Để tối ưu hóa tốc độ xử lý trên ma trận lớn, thay vì dùng vòng lặp, đoạn code sử dụng hằng đẳng thức vector hóa:
    $$(A - B)^2 = A^2 + B^2 - 2AB$$
    
    Công thức triển khai:
    $$Distance = \sqrt{X_{new}^2 + X_{train}^2 - 2(X_{new} \cdot X_{train}^T)}$$
    
    Trong đó, tích vô hướng ($2AB$) đóng vai trò cốt lõi giúp tận dụng sức mạnh tính toán ma trận của NumPy.

* **Tìm kiếm láng giềng (Nearest Neighbor Search):**
    Sử dụng `np.argsort` để sắp xếp khoảng cách từ nhỏ đến lớn, sau đó trích xuất $k$ chỉ số (index) có khoảng cách nhỏ nhất tương ứng với $k$ láng giềng gần nhất.

* **Bầu chọn đa số (Majority Voting):**
    Xác định nhãn của dữ liệu mới dựa trên nguyên tắc "thiểu số phục tùng đa số" trong tập $k$ láng giềng. Sử dụng `np.bincount` để đếm tần suất xuất hiện của các nhãn và `argmax` để chọn ra nhãn có số phiếu cao nhất.

---

### III.3.3 Thuật toán Gaussian Naive Bayes

* **Định nghĩa**
Đây là thuật toán phân loại dựa trên **Định lý Bayes** với giả định rằng các đặc trưng (features) độc lập với nhau và tuân theo phân phối chuẩn (Gaussian distribution).

* **Quy trình xây dựng mô hình**

**1. Huấn luyện (Training Phase - Thống kê dữ liệu)**
Thay vì tối ưu hóa hàm mất mát, mô hình "học" bằng cách tính toán trực tiếp các tham số thống kê cho từng lớp dữ liệu:
* **Tham số phân phối:** Tính giá trị trung bình ($\mu$) và phương sai ($\sigma^2$) cho từng đặc trưng của mỗi lớp.
* **Xác suất tiên nghiệm ($P(Class)$):** Tính tỷ lệ xuất hiện của mỗi lớp trong tập huấn luyện.
* **Ổn định số học:** Cộng thêm một hằng số cực nhỏ (`1e-9`) vào phương sai để làm mượt (smoothing), ngăn chặn lỗi chia cho 0.

**2. Dự đoán (Prediction Phase - Log-Likelihood)**
Để tránh lỗi tràn số dưới (numerical underflow) khi nhân nhiều giá trị xác suất nhỏ, thuật toán thực hiện tính toán trong không gian Logarit kết hợp với kỹ thuật **Broadcasting** của NumPy để xử lý song song trên ma trận 3 chiều (Samples x Classes x Features):

* **Tính Log-Likelihood:**
  Độ "khớp" của dữ liệu mới với phân phối chuẩn của từng lớp được tính theo công thức:
  $$\log P(x|c) = -\frac{1}{2} \sum \left( \log(2\pi\sigma_c^2) + \frac{(x - \mu_c)^2}{\sigma_c^2} \right)$$
  *(Bao gồm tổng của phần log mẫu số chuẩn hóa và khoảng cách Mahalanobis bình phương).*

* **Quyết định phân loại (Maximum A Posteriori):**
  Áp dụng định lý Bayes bằng cách cộng Log-likelihood với Log xác suất tiên nghiệm và chọn lớp có giá trị lớn nhất:
  $$\hat{y} = \text{argmax} \left( \log P(x|c) + \log P(c) \right)$$

Dưới đây là phần tóm tắt ngắn gọn, súc tích nhưng bao hàm đầy đủ các ý tưởng kỹ thuật quan trọng bạn đã cung cấp, được định dạng chuẩn để đưa vào mục III.4 của README:


### III.3.4 Chiến lược Đánh giá & Kiểm thử Mô hình

Để đảm bảo kết quả đánh giá khách quan và tối ưu hóa hiệu năng mô hình, quy trình kiểm thử được xây dựng chặt chẽ thông qua 3 thành phần chính:

#### III.3.4.1 Kỹ thuật K-Fold Cross-Validation
Thay vì chỉ chia dữ liệu một lần (Train/Test split truyền thống), ta áp dụng **K-Fold** để giảm thiểu phương sai và đánh giá độ ổn định của mô hình:
1.  **Xáo trộn (Shuffle):** Đảm bảo tính ngẫu nhiên, phá vỡ thứ tự sắp xếp gốc của dữ liệu.
2.  **Chia & Xoay vòng:** Dữ liệu được chia thành $k$ phần. Quy trình lặp $k$ lần, mỗi lần chọn một phần làm tập Test (Validation) và phần còn lại làm tập Train.
3.  **Lợi ích:** Đảm bảo 100% dữ liệu đều được kiểm thử và mô hình không bị "học vẹt" (overfitting) trên một tập mẫu cụ thể.

#### III.4.2 Quy trình vận hành (`evaluate_models`)
Hàm quản lý luồng đánh giá tuân thủ nghiêm ngặt nguyên tắc **chống rò rỉ dữ liệu (Data Leakage Prevention)**:
* **Bước 1 - Tách dữ liệu:** Tại mỗi vòng lặp K-Fold, dữ liệu được chia thành `Train_fold` và `Test_fold`.
* **Bước 2 - Xử lý mất cân bằng:** Hàm `oversample_minority` **CHỈ được áp dụng trên `Train_fold`**. Tập `Test_fold` được giữ nguyên bản để phản ánh đúng thực tế.
* **Bước 3 - Tổng hợp:** Kết quả của $k$ lần chạy được tính trung bình (`np.mean`) để đưa ra con số hiệu năng cuối cùng đáng tin cậy nhất.

#### III.3.4.3 Các chỉ số đánh giá (Metrics)
Dựa trên **Ma trận nhầm lẫn (Confusion Matrix)** với các yếu tố TP (Dương tính thật), FP (Dương tính giả) và FN (Âm tính giả), hiệu năng mô hình được đo lường qua:

* **Precision (Độ chính xác dự báo dương):** Tỉ lệ dự đoán đúng trong các trường hợp mô hình báo là Positive.
    $$P = \frac{TP}{TP + FP}$$
* **Recall (Độ nhạy):** Khả năng mô hình phát hiện được bao nhiêu % trường hợp Positive thực tế.
    $$R = \frac{TP}{TP + FN}$$
* **F1-Score:** Trung bình điều hòa giữa Precision và Recall, là chỉ số quan trọng nhất để đánh giá sự cân bằng của mô hình trên dữ liệu lệch.
    $$F1 = 2 \times \frac{P \times R}{P + R}$$

# **Installation & Setup**

```bash
git clone https://github.com/AnhTtis/Customer-Analysis
cd Customer-Analysis
pip install -r requirements.txt
```

---

# **Usage**

## Chạy từng notebook

* `01_data_exploration.ipynb` — phân tích dữ liệu
* `02_preprocessing.ipynb` — xử lý dữ liệu
* `03_modelling.ipynb` — huấn luyện & đánh giá mô hình
---

# 📈 **Results**

### Metrics đạt được (tùy mô hình)

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

### Trực quan hóa

* Biểu đồ phân phối churn
* Ma trận tương quan
* Histogram của các biến quan trọng
* Biểu đồ ROC

### So sánh mô hình

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
|── notebooks/
    ├── 01_data_exploration.ipynb
    ├── 02_preprocessing.ipynb
    └── 03_modelling.ipynb

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
