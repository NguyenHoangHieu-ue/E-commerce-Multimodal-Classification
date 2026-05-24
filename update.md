# 🚀 HƯỚNG DẪN SETUP NHANH (CHO MÁY MỚI)

Để tiết kiệm thời gian khi chuyển máy, bạn chỉ cần cài sẵn **Python 3.9+** và **Gemini CLI**. Sau đó, copy/paste hoặc bảo AI chạy các lệnh sau:

### 1. Tạo môi trường ảo (Khuyên dùng)
```powershell
python -m venv venv; .\venv\Scripts\activate
```

### 2. Cài đặt thư viện (Tự động nhận diện GPU/CPU)
*   **Nếu máy CÓ GPU NVIDIA (Để train nhanh):**
    ```powershell
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121; pip install -r requirements.txt
    ```
*   **Nếu máy CHỈ CÓ CPU (Để chạy Demo/Quick Eval):**
    ```powershell
    pip install -r requirements.txt
    ```

---

# 🛠 Danh sách Cần Cải thiện & Sửa lỗi (Backlog)

Tài liệu này ghi lại các vấn đề phát hiện trong quá trình test và kế hoạch tối ưu hóa dự án cho lần cập nhật tới.

---

## 🛑 1. Sửa lỗi Gán nhãn Dữ liệu (Heuristics Fix) - **ƯU TIÊN CAO**
- [ ] **Tinh chỉnh từ khóa ngành Thực phẩm:** Loại bỏ hoặc thêm điều kiện loại trừ cho các từ như `milk`, `susu`, `collagen`. Hiện tại các từ này đang khiến mỹ phẩm (sữa rửa mặt, kem dưỡng) bị nhầm sang ngành Thực phẩm & Đồ uống.
- [ ] **Bổ sung từ khóa phủ định:** Cập nhật script `Rules/expand_silver.py` để nếu tiêu đề chứa `soap`, `toner`, `serum` thì tuyệt đối không gán vào Thực phẩm kể cả khi có từ `milk`.

## 🧠 2. Huấn luyện lại Mô hình (Model Retraining)
- [ ] **Chạy lại Pipeline:** Sau khi sửa Heuristics, cần chạy lại `src/expand_silver.py` để tạo `docs/silver_dataset.csv` sạch hơn.
- [ ] **Huấn luyện Advanced V3:** Chạy lại `src/train_full.py` trên GPU với dữ liệu mới để xóa bỏ các "định kiến" sai lầm hiện tại của mô hình.

## ⚡ 3. Tối ưu hóa Hiệu năng & Triển khai
- [ ] **Giảm độ trễ (Latency):** Nghiên cứu sử dụng **DistilBERT** thay cho BERT-base để ứng dụng Web load nhanh hơn và tốn ít RAM hơn trên Cloud.
- [ ] **Quantization:** Thử nghiệm kỹ thuật nén mô hình (weight quantization) để giảm dung lượng file `.pth` (hiện tại đang quá nặng ~800MB).

## 📊 4. Đánh giá & Phân tích
- [ ] **Mở rộng Gold Dataset:** Tăng số lượng mẫu gán nhãn thủ công lên 500-1000 mẫu để có cái nhìn chính xác nhất về độ hội tụ.
- [ ] **Xử lý đa ngôn ngữ:** Cải thiện hàm `clean_text` trong `app.py` để lọc bỏ thêm các stopword tiếng Indonesia đặc thù.

---
*Ghi chú: File này được cập nhật để thay thế báo cáo cũ, phục vụ cho giai đoạn bảo trì và nâng cấp.*
