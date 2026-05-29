import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from transformers import AutoTokenizer, AutoModel
import os
import json
import pandas as pd
import re

# --- CẤU HÌNH ---
BERT_MODEL_NAME = 'bert-base-multilingual-cased'
NUM_CLASSES = 10
CLASS_NAMES = [
    'Gia dụng', 'Mẹ & Bé', 'Nhà cửa & Đời sống', 'Phụ kiện thời trang', 
    'Sắc đẹp', 'Sức khỏe', 'Thể thao & Dã ngoại', 'Thời trang', 
    'Thực phẩm & Đồ uống', 'Điện tử'
]

# --- Danh sach Stopwords tieng Viet co ban ---
STOPWORDS = ["và", "của", "là", "có", "cho", "trong", "các", "với", "được", "như", "cho", "đã", "này", "khi", "mà", "về", "tại", "những"]

def clean_text(text):
    """Lam sach van ban: viet thuong, bo ky tu dac biet va stopword."""
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)

# --- ĐỊNH NGHĨA MÔ HÌNH ---

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(32, 256)
        self.fc2 = nn.Linear(256, num_classes)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.avgpool(x)
        x = x.view(-1, 32)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class MultimodalModel(nn.Module):
    def __init__(self, num_classes):
        super(MultimodalModel, self).__init__()
        resnet = models.resnet50(weights=None)
        self.image_branch = nn.Sequential(*list(resnet.children())[:-1])
        self.text_branch = AutoModel.from_pretrained(BERT_MODEL_NAME)
        self.fusion_dim = 2048 + 768
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, num_classes)
        )
    def forward(self, image, input_ids, attention_mask):
        img_features = self.image_branch(image).view(image.size(0), -1)
        text_outputs = self.text_branch(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]
        combined = torch.cat((img_features, text_features), dim=1)
        return self.classifier(combined)

# --- HÀM LOAD MÔ HÌNH ---

@st.cache_resource
def load_models():
    device = torch.device('cpu')
    errors = []
    
    # 1. Load Baseline CNN
    cnn = SimpleCNN(NUM_CLASSES)
    cnn_path = 'src/baseline_cnn.pth'
    if os.path.exists(cnn_path):
        try:
            # Kiểm tra kích thước file (LFS pointer thường < 1KB)
            if os.path.getsize(cnn_path) < 1024:
                errors.append(f"⚠️ File {cnn_path} hiện tại chỉ là link Git LFS (quá nhỏ). Vui lòng kiểm tra lại cấu hình Git LFS.")
            else:
                cnn.load_state_dict(torch.load(cnn_path, map_location=device))
                cnn.eval()
        except Exception as e:
            errors.append(f"❌ Lỗi load Baseline CNN: {str(e)}")
    else:
        errors.append(f"❓ Không tìm thấy file: {cnn_path}")

    # 2. Load Multimodal Model
    multi = MultimodalModel(NUM_CLASSES)
    multi_path = 'best_multimodal_model.pth'
    if os.path.exists(multi_path):
        try:
            if os.path.getsize(multi_path) < 1024:
                errors.append(f"⚠️ File {multi_path} hiện tại chỉ là link Git LFS. Bạn cần 'git lfs pull' hoặc kiểm tra dung lượng trên GitHub.")
            else:
                # Load với strict=False để linh hoạt hơn, nhưng kiểm tra tỷ lệ khớp
                state_dict = torch.load(multi_path, map_location=device)
                
                # Loại bỏ prefix 'module.' nếu mô hình được huấn luyện bằng DataParallel
                new_state_dict = {}
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith('module.') else k
                    new_state_dict[name] = v
                
                missing, unexpected = multi.load_state_dict(new_state_dict, strict=False)
                
                # Tính toán mức độ khớp
                total_params = len(multi.state_dict())
                matched_params = total_params - len(missing)
                match_rate = (matched_params / total_params) * 100
                
                if match_rate < 50:
                    errors.append(f"❌ Cảnh báo: Mô hình chỉ khớp {match_rate:.1f}%. Có thể file .pth này thuộc về kiến trúc khác (ví dụ ResNet18 thay vì ResNet50).")
                elif missing:
                    st.warning(f"🔔 Mô hình load thành công {match_rate:.1f}%. Thiếu {len(missing)} tham số (có thể do phiên bản thư viện khác nhau).")
                
                multi.eval()
        except Exception as e:
            errors.append(f"❌ Lỗi load Multimodal Model: {str(e)}")
    else:
        errors.append(f"❓ Không tìm thấy file: {multi_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    
    # Hiển thị thông báo lỗi lên giao diện nếu có
    if errors:
        for err in errors:
            st.error(err)
            
    return cnn, multi, tokenizer

# --- XỬ LÝ ẢNH ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- GIAO DIỆN STREAMLIT ---

st.set_page_config(page_title="Shopee Multimodal Demo", layout="wide")

st.title("🛍️ Shopee Product Classification")

tab1, tab2 = st.tabs(["🔍 Dự đoán sản phẩm", "📊 Phân tích mô hình"])

with tab1:
    st.markdown("""
    So sánh giữa mô hình **Baseline (Chỉ dùng ảnh)** và **Advanced (Kết hợp Ảnh + Tiêu đề)**.
    """)

    # Sidebar: Nhập liệu
    st.sidebar.header("Nhập thông tin sản phẩm")
    uploaded_file = st.sidebar.file_uploader("Chọn ảnh sản phẩm...", type=["jpg", "jpeg", "png"])
    product_title = st.sidebar.text_input("Nhập tiêu đề sản phẩm:", placeholder="Ví dụ: Áo thun nam tay ngắn...")

    if uploaded_file is not None and product_title != "":
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sản phẩm đầu vào")
            st.image(image, caption=f"Tiêu đề: {product_title}", use_container_width=True)

        with col2:
            st.subheader("Kết quả dự đoán")
            
            with st.spinner('Đang tính toán...'):
                cnn_model, multi_model, tokenizer = load_models()

                cleaned_title = clean_text(product_title)
                img_tensor = transform(image).unsqueeze(0)
                inputs = tokenizer(cleaned_title, padding='max_length', truncation=True, max_length=64, return_tensors='pt')
                
                with torch.no_grad():
                    cnn_out = cnn_model(img_tensor)
                    cnn_probs = F.softmax(cnn_out, dim=1)[0]
                    cnn_sorted_val, cnn_sorted_idx = torch.sort(cnn_probs, descending=True)
                
                with torch.no_grad():
                    multi_out = multi_model(img_tensor, inputs['input_ids'], inputs['attention_mask'])
                    multi_probs = F.softmax(multi_out, dim=1)[0]
                    multi_sorted_val, multi_sorted_idx = torch.sort(multi_probs, descending=True)

                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    st.info("**Baseline (CNN)**")
                    st.metric("Dự đoán chính", CLASS_NAMES[cnn_sorted_idx[0]])
                    st.write("**Xác suất chi tiết:**")
                    for i in range(len(CLASS_NAMES)):
                        st.write(f"- {CLASS_NAMES[cnn_sorted_idx[i]]}: {cnn_sorted_val[i]*100:.2f}%")

                with res_col2:
                    st.success("**Advanced (Multimodal)**")
                    st.metric("Dự đoán chính", CLASS_NAMES[multi_sorted_idx[0]])
                    st.write("**Xác suất chi tiết:**")
                    for i in range(len(CLASS_NAMES)):
                        st.write(f"- {CLASS_NAMES[multi_sorted_idx[i]]}: {multi_sorted_val[i]*100:.2f}%")

                st.write("---")
                st.write("**Biểu đồ phân bổ xác suất (Multimodal):**")
                chart_data = {CLASS_NAMES[i]: float(multi_probs[i]) for i in range(len(CLASS_NAMES))}
                st.bar_chart(chart_data)
    else:
        st.info("Vui lòng tải ảnh lên và nhập tiêu đề sản phẩm ở thanh bên trái để bắt đầu dự đoán.")

with tab2:
    st.header("📈 So sánh hiệu năng huấn luyện")
    
    history_path = 'docs/training_history.json'
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        # Chuẩn bị dữ liệu cho biểu đồ
        metrics = st.selectbox("Chọn chỉ số so sánh:", ["Accuracy (Validation)", "Loss (Validation)"])
        
        key = 'va' if metrics == "Accuracy (Validation)" else 'vl'
        
        df_plot = pd.DataFrame({
            'Baseline CNN': history['baseline_cnn'][key],
            'Baseline LSTM': history['baseline_lstm'][key],
            'Advanced Multimodal': history['advanced_multimodal'][key]
        })
        
        st.line_chart(df_plot)
        
        st.markdown("""
        **Nhận xét:**
        - Mô hình **Advanced Multimodal** thường có độ chính xác cao hơn nhờ kết hợp thông tin từ cả hai nguồn.
        - **Baseline LSTM** có thể đạt kết quả tốt nếu tiêu đề chứa nhiều từ khóa đặc trưng.
        - **Baseline CNN** gặp khó khăn hơn khi chỉ dựa vào hình ảnh thô.
        """)
    else:
        st.warning("Không tìm thấy file lịch sử huấn luyện. Vui lòng chạy toàn bộ Notebook để tạo file JSON.")

st.markdown("---")
st.caption("Đồ án môn học Data Mining - Nhóm 09. Sử dụng BERT & ResNet50.")