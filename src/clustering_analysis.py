import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torchvision import models, transforms
from PIL import Image
from transformers import AutoTokenizer, AutoModel
import os
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# --- CONFIG ---
BERT_MODEL_NAME = 'bert-base-multilingual-cased'
NUM_CLASSES = 10
CLASS_NAMES = [
    'Gia dụng', 'Mẹ & Bé', 'Nhà cửa & Đời sống', 'Phụ kiện thời trang', 
    'Sắc đẹp', 'Sức khỏe', 'Thể thao & Dã ngoại', 'Thời trang', 
    'Thực phẩm & Đồ uống', 'Điện tử'
]
MODEL_PATH = 'best_multimodal_model.pth' 
SILVER_CSV = 'docs/silver_dataset.csv'
IMAGE_DIR = 'shopee-product-matching/train_images/'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAMPLE_SIZE = 500  # Giảm xuống 500 để chạy nhanh trên CPU

# --- MODEL DEFINITION (Khớp với app.py) ---
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
        
        # Trả về cả logits và vector fusion để phân tích clustering
        logits = self.classifier(combined)
        return logits, combined

# --- DATASET ---
class ShopeeDataset(Dataset):
    def __init__(self, df, transform, tokenizer):
        self.df = df
        self.transform = transform
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(IMAGE_DIR, row['image'])
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            # Nếu thiếu ảnh, tạo ảnh trống
            image = torch.zeros(3, 224, 224)
            
        inputs = self.tokenizer(row['title'], padding='max_length', truncation=True, max_length=64, return_tensors='pt')
        return image, inputs['input_ids'].squeeze(0), inputs['attention_mask'].squeeze(0), row['category']

def run_clustering():
    print(f"--- Bắt đầu phân tích Clustering trên {DEVICE} ---")
    
    # 1. Load Data
    if not os.path.exists(SILVER_CSV):
        print(f"❌ Lỗi: Không tìm thấy file {SILVER_CSV}")
        return
        
    df = pd.read_csv(SILVER_CSV)
    
    if len(df) > SAMPLE_SIZE:
        df = df.sample(SAMPLE_SIZE, random_state=42)
    
    # 2. Load Model
    model = MultimodalModel(NUM_CLASSES).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        print(f"📦 Đang load trọng số từ {MODEL_PATH}...")
        try:
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
            new_state_dict = { (k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items() }
            model.load_state_dict(new_state_dict, strict=False)
            print("✅ Load model thành công.")
        except Exception as e:
            print(f"❌ Lỗi load model: {e}")
    else:
        print(f"⚠️ Cảnh báo: Không tìm thấy {MODEL_PATH}. Script sẽ chạy với trọng số ngẫu nhiên (chỉ để test code).")
    
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = ShopeeDataset(df, transform, tokenizer)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    # 3. Trích xuất đặc trưng
    all_features = []
    all_labels = []
    
    print("🚀 Đang trích xuất đặc trưng (Embeddings)...")
    with torch.no_grad():
        for imgs, ids, masks, labels in tqdm(loader):
            imgs, ids, masks = imgs.to(DEVICE), ids.to(DEVICE), masks.to(DEVICE)
            _, features = model(imgs, ids, masks)
            all_features.append(features.cpu().numpy())
            all_labels.extend(labels)
            
    all_features = np.vstack(all_features)
    
    # 4. Clustering (K-Means)
    print("📈 Đang thực hiện gom cụm (K-Means)...")
    le = LabelEncoder()
    true_label_indices = le.fit_transform(all_labels)
    n_clusters = len(np.unique(true_label_indices))
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(all_features)
    
    # 5. Tính toán chỉ số Silhouette
    sil_score = silhouette_score(all_features, true_label_indices)
    print(f"📊 Silhouette Score (Dựa trên nhãn thật): {sil_score:.4f}")
    
    # 6. Trực quan hóa (t-SNE)
    print("🎨 Đang giảm chiều dữ liệu với t-SNE để vẽ biểu đồ...")
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    features_2d = tsne.fit_transform(all_features)
    
    # Vẽ biểu đồ
    plt.figure(figsize=(14, 10))
    sns.scatterplot(
        x=features_2d[:, 0], y=features_2d[:, 1], 
        hue=all_labels, 
        palette='tab10', 
        legend='full',
        alpha=0.8,
        s=100
    )
    plt.title(f"Trực quan hóa không gian đặc trưng Multimodal (t-SNE)\nSilhouette Score: {sil_score:.4f}", fontsize=15)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title="Ngành hàng")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    output_path = 'output/clustering_tsne.png'
    os.makedirs('output', exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"✅ Biểu đồ đã được lưu tại: {output_path}")

if __name__ == "__main__":
    run_clustering()
