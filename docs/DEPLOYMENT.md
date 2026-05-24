# Hướng dẫn Triển khai (Deployment Guide)

Tài liệu này hướng dẫn cách đưa ứng dụng **Shopee Product Classification** lên các nền tảng đám mây.

## 1. Chuẩn bị trước khi triển khai
- Đảm bảo file `app.py` nằm ở thư mục gốc.
- File `requirements.txt` đã bao gồm đầy đủ thư viện (đã tối ưu cho CPU).
- Các file model `.pth` (đặc biệt là `best_multimodal_model.pth`) đã được đẩy lên GitHub bằng **Git LFS**.

## 2. Triển khai lên Streamlit Community Cloud (Khuyên dùng)
Đây là cách nhanh nhất và hoàn toàn miễn phí.

1. Truy cập [share.streamlit.io](https://share.streamlit.io/).
2. Đăng nhập bằng tài khoản GitHub của bạn.
3. Nhấn **"New app"**.
4. Chọn repository: `E-commerce-Multimodal-Classification`.
5. Chọn nhánh: `main`.
6. Chọn file chính: `app.py`.
7. Nhấn **"Deploy!"**.

*Lưu ý: Do model nặng (~800MB) và BERT khá tốn RAM, ứng dụng có thể mất 2-3 phút để khởi tạo lần đầu.*

## 3. Triển khai lên HuggingFace Spaces
HuggingFace hỗ trợ tốt cho các ứng dụng Machine Learning.

1. Tạo một **Space** mới trên HuggingFace.
2. Chọn SDK là **Streamlit**.
3. Kết nối với GitHub repository của bạn hoặc upload trực tiếp.
4. HuggingFace tự động hỗ trợ Git LFS nên việc tải model sẽ diễn ra suôn sẻ.

## 4. Cấu hình Tài nguyên (Memory)
- **CPU:** Ứng dụng chạy tốt trên CPU (đã cấu hình trong `requirements.txt`).
- **RAM:** Cần tối thiểu 2GB RAM để load đồng thời ResNet50 và BERT. Nếu bị crash (Out of Memory), hãy kiểm tra logs trên nền tảng triển khai.

---
*Cập nhật bởi: Gemini CLI - 2026-05-24*
