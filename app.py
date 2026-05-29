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
        # FIX CỐ ĐỊNH: Dùng ResNet50 (2048) + BERT (768) = 2816
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
    cnn_loaded = False
    if os.path.exists('src/baseline_cnn.pth'):
        try:
            if os.path.getsize('src/baseline_cnn.pth') > 1000:
                cnn.load_state_dict(torch.load('src/baseline_cnn.pth', map_location=device), strict=False)
                cnn_loaded = True
        except: pass
    cnn.eval()

    # 2. Load Multimodal
    multi = MultimodalModel(NUM_CLASSES)
    multi_loaded = False
    path = 'best_multimodal_model.pth'
    if os.path.exists(path):
        try:
            if os.path.getsize(path) < 1000000:
                st.error(f"❌ File {path} trên GitHub hiện tại chỉ là link Git LFS (không chứa dữ liệu thật).")
            else:
                state_dict = torch.load(path, map_location=device)
                new_state_dict = { (k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items() }
                multi.load_state_dict(new_state_dict, strict=False)
                multi_loaded = True
        except Exception as e:
            st.error(f"Lỗi load mô hình Multimodal: {e}")
    else:
        st.error(f"❌ KHÔNG TÌM THẤY FILE: {path}")
        st.write("📂 **Các file hiện có trong thư mục:**")
        st.write(os.listdir('.'))
    
    multi.eval()
    
    if not multi_loaded:
        st.warning("⚠️ Cảnh báo: Mô hình Advanced chưa được load trọng số thật, kết quả dự đoán sẽ không chính xác.")

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
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
st.sidebar.success("Phiên bản: FINAL DEPLOY (ResNet50)")

uploaded_file = st.sidebar.file_uploader("Chọn ảnh sản phẩm...", type=["jpg", "jpeg", "png"])
product_title = st.sidebar.text_input("Nhập tiêu đề sản phẩm:")

if uploaded_file and product_title:
    image = Image.open(uploaded_file).convert('RGB')
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Đầu vào", use_container_width=True)
    with col2:
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

            st.info(f"**Baseline (Chỉ ảnh):** {cnn_pred}")
            st.success(f"**Advanced (Ảnh + Chữ):** {multi_pred}")
            
            st.write("---")
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                st.write("**Xác suất Baseline (CNN):**")
                st.bar_chart({CLASS_NAMES[i]: float(cnn_probs[i]) for i in range(len(CLASS_NAMES))})
            with chart_col2:
                st.write("**Xác suất Multimodal (Advanced):**")
                st.bar_chart({CLASS_NAMES[i]: float(multi_probs[i]) for i in range(len(CLASS_NAMES))})
else:
    st.info("Nhập ảnh và tiêu đề để bắt đầu.")
