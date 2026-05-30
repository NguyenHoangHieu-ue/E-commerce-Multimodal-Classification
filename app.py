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
import matplotlib.pyplot as plt
import seaborn as sns

# --- CẤU HÌNH ---
BERT_MODEL_NAME = 'bert-base-multilingual-cased'
NUM_CLASSES = 10
CLASS_NAMES = [
    'Gia dụng', 'Mẹ & Bé', 'Nhà cửa & Đời sống', 'Phụ kiện thời trang', 
    'Sắc đẹp', 'Sức khỏe', 'Thể thao & Dã ngoại', 'Thời trang', 
    'Thực phẩm & Đồ uống', 'Điện tử'
]
SILVER_CSV = 'docs/silver_dataset.csv'

# --- Danh sach Stopwords tieng Viet co ban ---
STOPWORDS = ["và", "của", "là", "có", "cho", "trong", "các", "with", "được", "như", "cho", "đã", "này", "khi", "mà", "về", "tại", "những"]

def clean_text(text):
    text = str(text)
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
            nn.Linear(self.fusion_dim, 512), 
            nn.ReLU(), 
            nn.Dropout(0.2), 
            nn.Linear(512, num_classes)
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
    
    # 1. Load Baseline CNN
    cnn = SimpleCNN(NUM_CLASSES)
    if os.path.exists('src/baseline_cnn.pth'):
        try:
            if os.path.getsize('src/baseline_cnn.pth') > 1000:
                cnn.load_state_dict(torch.load('src/baseline_cnn.pth', map_location=device), strict=False)
        except: pass
    cnn.eval()

    # 2. Load Multimodal
    multi = MultimodalModel(NUM_CLASSES)
    path = 'best_multimodal_model.pth'
    if os.path.exists(path):
        try:
            if os.path.getsize(path) > 1000000:
                state_dict = torch.load(path, map_location=device)
                new_state_dict = { (k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items() }
                multi.load_state_dict(new_state_dict, strict=False)
        except: pass
    multi.eval()

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    return cnn, multi, tokenizer

# --- XỬ LÝ ẢNH ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- GIAO DIỆN STREAMLIT ---
st.set_page_config(page_title="Shopee Multimodal Analysis", layout="wide", page_icon="🛍️")

st.sidebar.title("🛍️ Shopee Multimodal")
st.sidebar.markdown("Dự án phân loại sản phẩm đa phương thức kết hợp **ResNet50** và **BERT**.")
st.sidebar.divider()
st.sidebar.success("Phiên bản: Production v1.0")

# Tab navigation
tab1, tab2, tab3 = st.tabs(["🔍 Dự đoán sản phẩm", "📊 Kết quả & EDA", "🧬 Phân cụm (Clustering)"])

with tab1:
    st.header("🔍 Dự đoán ngành hàng sản phẩm")
    st.write("Tải lên một hình ảnh và nhập tiêu đề để xem mô hình phân loại sản phẩm vào ngành hàng nào.")
    
    col_input, col_res = st.columns([1, 2])
    
    with col_input:
        uploaded_file = st.file_uploader("Chọn ảnh sản phẩm...", type=["jpg", "jpeg", "png"])
        product_title = st.text_input("Nhập tiêu đề sản phẩm:", placeholder="Ví dụ: Áo thun nam cotton co giãn...")
        
    if uploaded_file and product_title:
        image = Image.open(uploaded_file).convert('RGB')
        with col_res:
            with st.spinner('Đang dự đoán...'):
                cnn_model, multi_model, tokenizer = load_models()
                img_tensor = transform(image).unsqueeze(0)
                inputs = tokenizer(clean_text(product_title), padding='max_length', truncation=True, max_length=64, return_tensors='pt')
                
                with torch.no_grad():
                    cnn_out = cnn_model(img_tensor)
                    multi_out = multi_model(img_tensor, inputs['input_ids'], inputs['attention_mask'])
                    
                    cnn_probs = F.softmax(cnn_out, dim=1)[0]
                    multi_probs = F.softmax(multi_out, dim=1)[0]
                    
                    cnn_pred = CLASS_NAMES[torch.argmax(cnn_probs)]
                    multi_pred = CLASS_NAMES[torch.argmax(multi_probs)]

                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.image(image, caption="Ảnh đầu vào", use_container_width=True)
                with res_col2:
                    st.info(f"**Baseline (Chỉ ảnh):** \n\n {cnn_pred}")
                    st.success(f"**Advanced (Ảnh + Chữ):** \n\n {multi_pred}")
                
                st.write("---")
                chart_col1, chart_col2 = st.columns(2)
                with chart_col1:
                    st.write("**Xác suất Baseline (CNN):**")
                    st.bar_chart({CLASS_NAMES[i]: float(cnn_probs[i]) for i in range(len(CLASS_NAMES))})
                with chart_col2:
                    st.write("**Xác suất Multimodal (Advanced):**")
                    st.bar_chart({CLASS_NAMES[i]: float(multi_probs[i]) for i in range(len(CLASS_NAMES))})
    else:
        with col_res:
            st.info("💡 **Gợi ý:** Bạn có thể lấy ảnh và tiêu đề từ trang Shopee để thử nghiệm độ chính xác của mô hình.")
            st.image("https://deo.shopeemobile.com/shopee/shopee-pcmall-live-sg/assets/ca5a12a11391d4d150190209e7471974.png", width=400)

with tab2:
    st.header("📊 Kết quả Thực nghiệm & EDA")
    
    # 1. Performance Section
    st.subheader("🏆 Hiệu năng mô hình (Validation)")
    col_metric1, col_metric2, col_metric3 = st.columns(3)
    col_metric1.metric("Baseline LSTM", "19.3%", delta="-43.9%")
    col_metric2.metric("Baseline CNN", "25.9%", delta="-37.3%")
    col_metric3.metric("Advanced Multimodal", "63.2%", delta="+37.3%")
    
    res_img_col1, res_img_col2 = st.columns(2)
    with res_img_col1:
        if os.path.exists('output/pic5.png'):
            st.image('output/pic5.png', caption="Đồ thị Accuracy trong quá trình huấn luyện", use_container_width=True)
    with res_img_col2:
        if os.path.exists('output/pic6.png'):
            st.image('output/pic6.png', caption="Confusion Matrix (Ma trận nhầm lẫn)", use_container_width=True)

    st.divider()

    # 2. EDA Section
    if os.path.exists(SILVER_CSV):
        df_silver = pd.read_csv(SILVER_CSV)
        
        st.subheader("📌 Thống kê tập dữ liệu Silver")
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        col_stat1.metric("Tổng số mẫu", len(df_silver))
        col_stat2.metric("Số ngành hàng", df_silver['category'].nunique())
        col_stat3.metric("Số nhóm nhãn", df_silver['label_group'].nunique())
        
        st.subheader("📈 Phân bố ngành hàng")
        if os.path.exists('output/histogram.png'):
            st.image('output/histogram.png', caption="Số lượng sản phẩm mỗi ngành hàng", use_container_width=True)
            
        st.subheader("📋 Mẫu dữ liệu huấn luyện")
        st.dataframe(df_silver[['title', 'category', 'image']].head(10), use_container_width=True)
    else:
        st.error("Không tìm thấy tệp silver_dataset.csv để hiển thị EDA.")

with tab3:
    st.header("🧬 Phân tích Phân cụm (Clustering)")
    st.write("""
    Phân cụm giúp chúng ta hiểu cách mô hình 'nhìn' dữ liệu trong không gian đặc trưng. 
    Các điểm gần nhau đại diện cho các sản phẩm mà mô hình cho là tương đồng về cả hình ảnh và nội dung văn bản.
    """)
    
    if os.path.exists('output/clustering_tsne.png'):
        st.image('output/clustering_tsne.png', caption="Trực quan hóa t-SNE trên vector đặc trưng Multimodal", use_container_width=True)
        
        st.markdown("""
        **Giải thích biểu đồ:**
        - Mỗi dấu chấm đại diện cho một sản phẩm trong tập dữ liệu.
        - Màu sắc đại diện cho nhãn ngành hàng thực tế của sản phẩm.
        - Không gian này được tạo ra bằng cách nối vector đặc trưng từ **ResNet50** (2048 chiều) và **BERT** (768 chiều), sau đó giảm chiều xuống 2D bằng thuật toán **t-SNE**.
        - Sự tập trung của các màu sắc thành từng cụm riêng biệt cho thấy mô hình đã học được cách phân biệt tốt các đặc trưng của từng ngành hàng.
        """)
    else:
        st.warning("⚠️ Không tìm thấy file trực quan hóa phân cụm (output/clustering_tsne.png).")
        st.info("Bạn có thể chạy script `src/clustering_analysis.py` để tạo biểu đồ này.")
