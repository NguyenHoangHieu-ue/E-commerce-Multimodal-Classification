# Phân loại sản phẩm Thương mại Điện tử Đa phương thức (Shopee)

Dự án nghiên cứu và triển khai hệ thống phân loại ngành hàng sản phẩm TMĐT tự động bằng cách kết hợp đặc trưng từ cả **Hình ảnh** và **Văn bản (Tiêu đề)**. Dự án sử dụng tập dữ liệu thực tế từ cuộc thi Shopee trên Kaggle.

## 🚀 Điểm nổi bật

- **Chiến lược dữ liệu thông minh**: Mở rộng tập huấn luyện (Silver Dataset) từ ~1.000 mẫu lên hơn **21.000 mẫu** bằng phương pháp gán nhãn tự động dựa trên từ khóa và nhóm sản phẩm.
- **Kiến trúc Multimodal**: Kết hợp sức mạnh của **BERT** (NLP) và **ResNet50** (Computer Vision) để hiểu sản phẩm một cách toàn diện.
- **Quy trình hoàn chỉnh**: Bao gồm đầy đủ các bước từ xử lý dữ liệu thô, EDA, huấn luyện mô hình đến đánh giá chi tiết bằng Confusion Matrix.

## 🧠 Các Mô hình triển khai

Dự án tập trung so sánh giữa hai cấp độ tiếp cận:

1. **Baseline V1 (Mô hình cơ bản)**:
   - **Ảnh**: Mạng CNN đơn giản (3 layers).
   - **Chữ**: Mạng LSTM xử lý chuỗi từ.
   - *Mục tiêu*: Đánh giá hiệu năng khi huấn luyện từ đầu trên tập dữ liệu nhỏ.

2. **Advanced V3 (Multimodal Fusion)**:
   - **Text Branch**: `bert-base-multilingual-cased` để xử lý tiêu đề đa ngôn ngữ (Việt, Anh, Indo).
   - **Image Branch**: `ResNet50` (Pre-trained) để trích xuất đặc trưng hình ảnh chuyên sâu.
   - **Fusion**: Kỹ thuật Concatenation nối các vector đặc trưng để đưa ra dự đoán cuối cùng.

## 📊 Dữ liệu (Dataset)

- **Nguồn**: [Kaggle - Shopee Price Match Guarantee](https://www.kaggle.com/c/shopee-product-matching/data).
- **Silver Dataset**: >21.000 mẫu gán nhãn tự động (dùng để huấn luyện).

## 📂 Cấu trúc thư mục

```text
.
├── docs/               # Chứa tập dữ liệu Silver/Gold và nhật ký thí nghiệm
├── notebooks/          # Notebook phục vụ EDA và thử nghiệm rời rạc
├── Rules/              # Quy định dự án và kịch bản gán nhãn dữ liệu
├── shopee-product-matching/ # Thư mục chứa ảnh và file CSV gốc (Kaggle)
├── src/                # Các script hỗ trợ (Xử lý dữ liệu, làm slide, báo cáo)
│   └── expand_silver.py# Script chính để mở rộng tập dữ liệu Silver
├── Shopee_Multimodal_Project.ipynb # File Notebook chính chứa toàn bộ Pipeline
└── README.md
```

## 🛠 Cài đặt & Sử dụng

### 1. Cài đặt môi trường

Yêu cầu Python 3.8+ và các thư viện sau:

```bash
pip install torch torchvision transformers pandas scikit-learn pillow wordcloud seaborn matplotlib
```

### 2. Mở rộng tập dữ liệu (Nếu cần gán nhãn lại)

Chạy script mở rộng tập Silver bằng các quy tắc từ khóa tiếng Việt:

```bash
py src/expand_silver.py
```

### 3. Huấn luyện và Đánh giá

Mở file `Shopee_Multimodal_Project.ipynb` bằng Jupyter Notebook hoặc Google Colab và chạy tuần tự các cell để:

- Khám phá dữ liệu (EDA).
- Huấn luyện Baseline V1.
- Huấn luyện Advanced Multimodal V3.
- Xem kết quả trực quan hóa và Confusion Matrix.
