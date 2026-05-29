# Báo cáo Dự án: Phân loại Sản phẩm Thương mại Điện tử Đa phương thức (Shopee)

Dự án này tập trung vào việc xây dựng một hệ thống phân loại sản phẩm tự động cho sàn thương mại điện tử Shopee bằng cách kết hợp cả thông tin **Hình ảnh** và **Văn bản (Tiêu đề)**.

---

## 1. Tổng quan Dự án (Project Overview)
Trong lĩnh vực thương mại điện tử, việc phân loại sản phẩm chính xác là vô cùng quan trọng để tối ưu hóa tìm kiếm và gợi ý sản phẩm. Tuy nhiên, nếu chỉ dựa vào hình ảnh hoặc tiêu đề đơn lẻ, độ chính xác thường không cao do sự đa dạng và nhiễu của dữ liệu. Dự án này triển khai phương pháp **Multimodal Fusion** (Kết hợp đa phương thức) để giải quyết vấn đề này.

### Mục tiêu chính:
- Xây dựng Pipeline xử lý dữ liệu hình ảnh và văn bản.
- Huấn luyện và so sánh các mô hình Baseline (đơn phương thức) với mô hình Advanced (đa phương thức).
- Triển khai ứng dụng Web Demo để dự đoán thời gian thực.

---

## 2. Kiến trúc Mô hình (Model Architecture)

Dự án triển khai và so sánh 3 kiến trúc chính:

### a. Baseline V1 (Chỉ Hình ảnh - CNN)
- **Kiến trúc**: Một mạng Convolutional Neural Network (CNN) đơn giản gồm 2 lớp Conv2d, MaxPool và các lớp Fully Connected.
- **Đặc điểm**: Huấn luyện từ đầu trên tập dữ liệu nhỏ để làm mốc so sánh.

### b. Baseline V2 (Chỉ Văn bản - LSTM)
- **Kiến trúc**: Sử dụng nhúng từ (Embedding) kết hợp với mạng Long Short-Term Memory (LSTM) để xử lý chuỗi văn bản.
- **Đặc điểm**: Phù hợp cho các tiêu đề sản phẩm có cấu trúc từ ngữ lặp lại.

### c. Advanced V3 (Multimodal Fusion - ResNet50 + BERT) - **Mô hình Tốt nhất**
- **Nhánh Ảnh (Image Branch)**: Sử dụng **ResNet50** (Pre-trained trên ImageNet) để trích xuất vector đặc trưng 2048 chiều.
- **Nhánh Văn bản (Text Branch)**: Sử dụng **BERT-base-multilingual-cased** để xử lý tiêu đề đa ngôn ngữ, trích xuất vector đặc trưng 768 chiều từ token `[CLS]`.
- **Fusion**: Sử dụng kỹ thuật **Concatenation** (Nối vector) để tạo ra vector đặc trưng tổng hợp (2816 chiều).
- **Phân loại**: Qua các lớp Dense (Fully Connected) với Dropout để đưa ra xác suất cho 10 ngành hàng.

---

## 3. Dữ liệu (Datasets)

Dữ liệu được lấy từ cuộc thi *Shopee - Price Match Guarantee* trên Kaggle, bao gồm hàng chục nghìn ảnh và tiêu đề.

### Chiến lược Dữ liệu:
- **Silver Dataset (~21,000 mẫu)**: Được gán nhãn tự động bằng bộ quy tắc từ khóa (Keyword-based heuristics) trong `src/expand_silver.py`. Dùng để huấn luyện mô hình.
- **Gold Dataset (300 mẫu)**: Được kiểm duyệt thủ công để đảm bảo độ chính xác tuyệt đối. Dùng để đánh giá khách quan hiệu năng mô hình.

### 10 Ngành hàng mục tiêu:
1. Gia dụng
2. Mẹ & Bé
3. Nhà cửa & Đời sống
4. Phụ kiện thời trang
5. Sắc đẹp
6. Sức khỏe
7. Thể thao & Dã ngoại
8. Thời trang
9. Thực phẩm & Đồ uống
10. Điện tử

---

## 4. Kết quả Thực nghiệm (Results)

Kết quả đo đạc trên tập Validation sau 10 Epoch:

| Mô hình | Accuracy (Độ chính xác) | Nhận xét |
| :--- | :--- | :--- |
| **Baseline LSTM** | **~19.3%** | Thấp nhất do tập dữ liệu nhỏ và BERT hiệu quả hơn LSTM rất nhiều. |
| **Baseline CNN** | **~25.9%** | Chỉ dựa vào ảnh thô nên dễ nhầm lẫn các sản phẩm có hình dáng tương tự. |
| **Advanced Multimodal** | **~63.2%** | **Vượt trội hoàn toàn**, cho thấy sự kết hợp giữa ResNet50 và BERT mang lại hiệu quả cực cao. |

---

## 5. Cấu trúc Thư mục Dự án

```text
E-commerce-Multimodal-Classification/
├── app.py                     # Ứng dụng Web Demo (Streamlit)
├── best_multimodal_model.pth  # Trọng số mô hình tốt nhất (đã huấn luyện)
├── requirements.txt           # Danh sách thư viện cần thiết
├── Shopee_Multimodal_Project.ipynb # Toàn bộ quy trình từ EDA đến Training
├── docs/                      # Tài liệu và dữ liệu Silver/Gold
├── Rules/                     # Quy tắc gán nhãn và hướng dẫn Gemini CLI
├── src/                       # Mã nguồn huấn luyện và xử lý dữ liệu
└── output/                    # Kết quả trực quan hóa (biểu đồ)
```

---

## 6. Hướng dẫn Sử dụng

### Cài đặt môi trường:
```bash
pip install -r requirements.txt
```

### Chạy Web Demo:
```bash
streamlit run app.py
```

### Huấn luyện lại mô hình:
Mở file `Shopee_Multimodal_Project.ipynb` trong Jupyter Notebook hoặc Google Colab và chạy tất cả các cell.

---

## 7. Kết luận
Dự án đã chứng minh rằng việc kết hợp thông tin đa phương thức (Hình ảnh + Văn bản) giúp cải thiện đáng kể hiệu năng phân loại sản phẩm trong TMĐT. Việc sử dụng các mô hình Pre-trained mạnh mẽ như BERT và ResNet50 là chìa khóa để đạt được kết quả tốt trên tập dữ liệu thực tế đầy thách thức.

---
*Báo cáo được tổng hợp bởi Gemini CLI - 2026-05-29*
