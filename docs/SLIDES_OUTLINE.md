# Dàn ý Slide Thuyết trình (Slide Outline)

*   **Slide 1: Tiêu đề**
    *   Tên đề tài: Phân loại Sản phẩm Shopee Đa phương thức (Multimodal Classification).
    *   Sinh viên thực hiện: Nguyễn Hoàng Hiếu.
    *   Mã sinh viên: 49.01.104.045.

*   **Slide 2: Bài toán & Động lực**
    *   Hình ảnh: Minh họa tiêu đề Shopee viết tắt (nhiễu) và ảnh sản phẩm thực tế.
    *   Nội dung: Phân loại đơn phương thức (chỉ ảnh hoặc chỉ chữ) dễ bị sai do dữ liệu TMĐT phức tạp. Cần giải pháp Đa phương thức (Multimodal).

*   **Slide 3: Bộ dữ liệu (Dataset)**
    *   Hình ảnh: `histogram.png` (Phân phối lớp) và `pic3.png` (Ảnh mẫu).
    *   Nội dung: 34,000 mẫu gốc. Xây dựng tập **Silver Dataset** (21,000 mẫu) bằng Keyword Heuristics để huấn luyện mô hình sâu.

*   **Slide 4: Kiến trúc Mô hình (Core Architecture)**
    *   Hình ảnh: Sơ đồ [Ảnh -> ResNet50] + [Chữ -> BERT] -> Concatenation -> Classifier.
    *   Key: Sử dụng **Intermediate Fusion** để kết hợp đặc trưng sâu của cả hai nguồn thông tin.

*   **Slide 5: Kết quả thực nghiệm**
    *   Hình ảnh: `pic5.png` (Accuracy curves) và `pic6.png` (Confusion Matrix).
    *   Nội dung: So sánh Baseline (~20-25%) với Multimodal (**63.2%**). Sự cải thiện vượt bậc.

*   **Slide 6: Kết luận & Demo**
    *   Hình ảnh: Ảnh chụp màn hình ứng dụng Streamlit (Web Demo).
    *   Nội dung: Mô hình có khả năng ứng dụng thực tế. Sẵn sàng cho việc mở rộng quy mô.
