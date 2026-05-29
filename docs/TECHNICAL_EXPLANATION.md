# Tài liệu Giải thích Kỹ thuật & Câu hỏi Phản biện (FAQ)

### 1. Tại sao phải xây dựng "Silver Dataset"?
*   **Vấn đề:** Dữ liệu gốc từ Kaggle chỉ phục vụ bài toán tìm sản phẩm giống nhau (Matching), không có nhãn ngành hàng (Category).
*   **Giải pháp:** Sử dụng bộ quy tắc từ khóa (Keyword Heuristics) trong `src/expand_silver.py`. Ví dụ: nếu tiêu đề có "điện thoại, laptop" -> gán nhãn "Điện tử".
*   **Giá trị:** Tạo ra lượng dữ liệu đủ lớn (21,000 mẫu) để các mô hình lớn như ResNet50 và BERT có thể học được (convergence).

### 2. Tại sao chọn ResNet50 thay vì các mạng khác?
*   **Kiến trúc:** ResNet50 sử dụng kết nối tắt (shortcut connections) giúp giải quyết vấn đề Vanishing Gradient trong các mạng sâu.
*   **Hiệu quả:** Nó cân bằng tốt giữa độ chính xác và tốc độ tính toán. Em sử dụng kỹ thuật Transfer Learning với trọng số từ ImageNet để tận dụng khả năng nhận diện vật thể cơ bản có sẵn.

### 3. Ưu điểm của BERT trong bài toán này là gì?
*   BERT có cơ chế **Self-Attention**, giúp nó hiểu ngữ nghĩa của từ dựa trên ngữ cảnh toàn câu, thay vì chỉ đọc từ trái sang phải như LSTM.
*   Em sử dụng bản **Multilingual** vì tiêu đề Shopee thường trộn lẫn tiếng Việt, Anh và Indonesia. Token `[CLS]` được lấy làm đại diện cho toàn bộ câu.

### 4. Cơ chế Fusion (Intermediate Fusion) hoạt động ra sao?
*   Em không kết hợp ở đầu ra (Late Fusion) mà kết hợp ở lớp đặc trưng (Feature level).
*   Vector ảnh (2048) và Vector chữ (768) được nối lại thành vector 2816 chiều. Việc này giúp mô hình học được mối quan hệ "chéo" giữa hai phương thức dữ liệu ngay trong quá trình huấn luyện.

### 5. Giải thích các con số trong Confusion Matrix (Hình 7)
*   Mô hình hoạt động tốt nhất ở ngành hàng **Điện tử** và **Sắc đẹp** do có nhiều từ khóa đặc trưng.
*   Nhầm lẫn nhiều nhất ở **Thời trang** và **Thể thao & Dã ngoại**. Nguyên nhân là do hình ảnh quần áo (áo thun, quần short) của hai ngành hàng này thường có đặc trưng thị giác rất giống nhau.

### 6. Tại sao kết quả đạt 63.2%? Cải thiện bằng cách nào?
*   Đây là kết quả trên dữ liệu "In-the-wild" (thực tế) cực kỳ nhiễu.
*   **Cải thiện:** Trong tương lai, có thể sử dụng cơ chế **Attention Fusion** (để mô hình tự quyết định lúc nào nên tin ảnh hơn, lúc nào nên tin chữ hơn) hoặc thực hiện **Fine-tuning** sâu hơn nếu có tài nguyên phần cứng mạnh hơn.
