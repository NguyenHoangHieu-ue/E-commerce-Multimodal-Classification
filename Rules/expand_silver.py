"""
Tên kịch bản: expand_silver.py
Mô tả: Mở rộng tập dữ liệu huấn luyện (Silver Dataset) cho dự án Shopee bằng cách 
       kết hợp phương pháp ánh xạ nhóm sản phẩm (label_group) và các quy tắc từ khóa (heuristics).
Tác giả: Gemini CLI
"""

import os
import pandas as pd
import numpy as np

# --- CẤU HÌNH ĐƯỜNG DẪN ---
TRAIN_CSV = 'shopee-product-matching/train.csv'
SILVER_CSV = 'docs/silver_dataset.csv'

# --- QUY TẮC TỪ KHÓA (HEURISTICS) ---
# Phân loại sản phẩm dựa trên các từ khóa đặc trưng trong tiêu đề
QUY_TAC_TU_KHOA = {
    'Thực phẩm & Đồ uống': [
        'milk', 'makanan', 'kurma', 'bumbu', 'sauce', 'noodle', 'snack', 'susu', 
        'coffee', 'kopi', 'cokelat', 'chocolate', 'drink'
    ],
    'Phụ kiện thời trang': [
        'sepatu', 'sandal', 'dompet', 'tangan', 'jam tangan', 'kacamata', 'cincin', 
        'kalung', 'gelang', 'anting', 'tas', 'selempang', 'bag'
    ],
    'Điện tử': [
        'speaker', 'charger', 'bluetooth', 'kabel', 'portable', 'micro', 'holder', 
        'usb', 'headphone', 'earphone', 'samsung', 'iphone', 'ipad', 'laptop', 
        'mouse', 'keyboard'
    ],
    'Nhà cửa & Đời sống': [
        'lampu', 'alat', 'botol', 'tape', 'putih', 'bening', 'motor', 'helm', 
        'kunci', 'sapu', 'rak', 'dispenser', 'pisau', 'panci', 'sprei'
    ],
    'Sức khỏe': [
        'obat', 'vitamin', 'herbal', 'tablet', 'ikan', 'madu', 'lambung', 'mata', 
        'lachel', 'masker medis'
    ],
    'Sắc đẹp': [
        'masker', 'serum', 'wajah', 'toner', 'mask', 'sabun', 'beauty', 'hair', 
        'lipstik', 'parfum', 'kosmetik', 'makeup', 'cream', 'acne', 'skincare'
    ],
    'Mẹ & Bé': [
        'bayi', 'anak', 'baby', 'mainan', 'puzzle', 'topi', 'bubur', 'kids', 
        'organik', 'susu formula', 'popok', 'sweety', 'diaper', 'pampers'
    ],
    'Thời trang': [
        'kaos', 'celana', 'baju', 'gamis', 'polos', 'kemeja', 'jaket', 'sweater', 
        'hoodie', 'dress', 'rok', 'hijab', 'jilbab'
    ],
    'Thể thao & Dã ngoại': [
        'sepeda', 'bell', 'olahraga', 'bola', 'jersey', 'sepatu bola', 'raket', 
        'tenda', 'camping', 'fishing', 'pancing'
    ],
    'Gia dụng': [
        'blender', 'kipas', 'kulkas', 'mesin cuci', 'setrika', 'mixer', 'oven', 
        'rice cooker', 'kompor'
    ]
}

def tai_danh_sach_anh_xa():
    """Trích xuất ánh xạ label_group -> category từ tập silver hiện tại."""
    anh_xa = {}
    
    if os.path.exists(SILVER_CSV):
        df_silver = pd.read_csv(SILVER_CSV)
        if 'label_group' in df_silver.columns and 'category' in df_silver.columns:
            anh_xa = dict(zip(df_silver['label_group'], df_silver['category']))
        
    return anh_xa

def phan_loai_san_pham(dong, anh_xa_da_biet):
    """Phân loại một dòng dữ liệu dựa trên ánh xạ cũ hoặc từ khóa."""
    # 1. Ưu tiên ánh xạ theo label_group nếu đã tồn tại
    if dong['label_group'] in anh_xa_da_biet:
        return anh_xa_da_biet[dong['label_group']]
    
    # 2. Sử dụng quy tắc từ khóa nếu là label_group mới
    tieu_de = str(dong['title']).lower()
    for nganh_hang, tu_khoa in QUY_TAC_TU_KHOA.items():
        if any(tk in tieu_de for tk in tu_khoa):
            return nganh_hang
            
    return 'Khác'

def thuc_thi_mo_rong():
    print("--- Bắt đầu quy trình mở rộng Silver Dataset ---")
    
    if not os.path.exists(TRAIN_CSV):
        print(f"Lỗi: Không tìm thấy file dữ liệu gốc tại {TRAIN_CSV}!")
        return

    # 1. Đọc dữ liệu
    print(f"Đang đọc file {TRAIN_CSV}...")
    df_goc = pd.read_csv(TRAIN_CSV)
    
    print("Đang tải danh sách ánh xạ nhãn hiện tại...")
    anh_xa_da_biet = tai_danh_sach_anh_xa()
    
    # 2. Thực hiện gán nhãn
    print("Đang tiến hành phân loại sản phẩm (vui lòng chờ trong giây lát)...")
    df_goc['category'] = df_goc.apply(
        lambda dong: phan_loai_san_pham(dong, anh_xa_da_biet), 
        axis=1
    )
    
    # 3. Lọc lấy các mẫu đã được gán nhãn thành công (loại bỏ 'Khác')
    df_mo_rong = df_goc[df_goc['category'] != 'Khác'].copy()
    
    # 4. Báo cáo kết quả
    print("\n--- KẾT QUẢ MỞ RỘNG ---")
    print(f"Tổng số dòng dữ liệu thô: {len(df_goc)}")
    print(f"Tổng số dòng Silver mới : {len(df_mo_rong)}")
    print("\nPhân bổ theo ngành hàng:")
    print(df_mo_rong['category'].value_counts())
    
    # 5. Lưu kết quả
    print(f"\nĐang lưu tập dữ liệu đã mở rộng vào {SILVER_CSV}...")
    df_mo_rong.to_csv(SILVER_CSV, index=False)
    
    print("\nHoàn thành: Tập dữ liệu đã sẵn sàng để huấn luyện.")

if __name__ == "__main__":
    thuc_thi_mo_rong()
