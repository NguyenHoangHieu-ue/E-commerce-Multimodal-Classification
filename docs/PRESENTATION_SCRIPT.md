# Kịch bản Thuyết trình 3 Phút (3-Minute Script)

**0:00 - 0:45 | Giới thiệu & Dữ liệu**
"Kính thưa Hội đồng, em tên là Nguyễn Hoàng Hiếu. Đề tài của em là Phân loại sản phẩm Shopee bằng phương pháp Đa phương thức. Trong thương mại điện tử, dữ liệu thường rất nhiễu: tiêu đề chứa nhiều từ rác, còn hình ảnh thì mờ hoặc trùng lặp. Để giải quyết, em đã sử dụng bộ dữ liệu 34,000 mẫu từ Kaggle. Thách thức lớn nhất là thiếu nhãn ngành hàng, nên em đã xây dựng thuật toán gán nhãn tự động dựa trên từ khóa tiếng Việt để tạo ra tập Silver Dataset hơn 21,000 mẫu cho 10 ngành hàng."

**0:45 - 2:00 | Kỹ thuật & Mô hình**
"Về mặt kỹ thuật, em đề xuất kiến trúc **Multimodal Fusion**. Nhánh hình ảnh sử dụng **ResNet50** để trích xuất đặc trưng không gian. Nhánh văn bản sử dụng **BERT Multilingual** để hiểu ngữ nghĩa tiêu đề đa ngôn ngữ. Điểm mấu chốt của đồ án là tầng Fusion, nơi em nối (concatenate) hai vector đặc trưng lại với nhau. Điều này cho phép các lớp phân loại phía sau học được sự tương quan trực tiếp giữa hình dáng sản phẩm và từ ngữ mô tả, giúp mô hình tự bổ khuyết thông tin cho nhau."

**2:00 - 3:00 | Kết quả & Kết luận**
"Kết quả thực nghiệm minh chứng cho sức mạnh của sự kết hợp: trong khi các mô hình đơn phương thức chỉ đạt độ chính xác khoảng 20-25%, mô hình Đa phương thức của em đã đạt tới **63.2%**. Em cũng đã đóng gói mô hình thành một ứng dụng Web Demo bằng Streamlit để người dùng có thể kiểm chứng kết quả dự đoán thời gian thực. Em xin cảm ơn Hội đồng đã lắng nghe và sẵn sàng nhận câu hỏi phản biện."
