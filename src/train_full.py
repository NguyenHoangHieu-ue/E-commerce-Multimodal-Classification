import os
import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import AutoTokenizer, AutoModel
import time

# --- CẤU HÌNH ---
TRAIN_CSV = 'shopee-product-matching/train.csv'
IMAGE_DIR = 'shopee-product-matching/train_images/'
SILVER_CSV = 'docs/silver_dataset.csv'
BERT_MODEL_NAME = 'bert-base-multilingual-cased'
IMG_SIZE = 224
MAX_LEN = 64
BATCH_SIZE = 32  # Tăng batch size vì dùng GPU
EPOCHS = 10
LR = 2e-5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Sử dụng thiết bị: {DEVICE}")

# --- TIỀN XỬ LÝ VĂN BẢN ---
STOPWORDS = ["và", "của", "là", "có", "cho", "trong", "các", "với", "được", "như", "cho", "đã", "này", "khi", "mà", "về", "tại", "những"]

def clean_text(text):
    """Lam sach van ban: viet thuong, bo ky tu dac biet va stopword."""
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)

# --- DATASET ---
class ShopeeMultimodalDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform, tokenizer, max_len=64):
        self.df = dataframe
        self.image_dir = image_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row['image'])
        image = self.transform(Image.open(image_path).convert('RGB'))
        
        inputs = self.tokenizer(
            clean_text(row['title']),
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        
        return {
            'image': image,
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'label': torch.tensor(row['label'], dtype=torch.long)
        }

# --- MÔ HÌNH (Advanced V3: ResNet50 + BERT) ---
class MultimodalModel(nn.Module):
    def __init__(self, num_classes, freeze_backbone=False):
        super(MultimodalModel, self).__init__()
        # Nhánh ảnh: ResNet50
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.image_branch = nn.Sequential(*list(resnet.children())[:-1])
        
        # Nhánh văn bản: BERT
        self.text_branch = AutoModel.from_pretrained(BERT_MODEL_NAME)
        
        if freeze_backbone:
            for param in self.image_branch.parameters(): param.requires_grad = False
            for param in self.text_branch.parameters(): param.requires_grad = False
            
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

# --- HUẤN LUYỆN VÀ ĐÁNH GIÁ ---
def train_model():
    print("--- Bắt đầu quy trình huấn luyện toàn bộ (GPU) ---")
    
    # Load dữ liệu Silver
    if not os.path.exists(SILVER_CSV):
        print(f"Lỗi: Không tìm thấy file {SILVER_CSV}")
        return

    df = pd.read_csv(SILVER_CSV)
    df = df[df['category'] != 'Khác'].reset_index(drop=True)
    
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['category'])
    num_classes = len(le.classes_)
    
    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df['label'], random_state=42
    )
    
    print(f"Số lượng mẫu: Train={len(train_df)}, Val={len(val_df)}")
    print(f"Số lượng class: {num_classes}")

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    
    train_ds = ShopeeMultimodalDataset(train_df, IMAGE_DIR, train_transform, tokenizer, MAX_LEN)
    val_ds = ShopeeMultimodalDataset(val_df, IMAGE_DIR, val_transform, tokenizer, MAX_LEN)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = MultimodalModel(num_classes).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch in train_loader:
            images = batch['image'].to(DEVICE)
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / total_train
        train_acc = correct_train / total_train

        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(DEVICE)
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['label'].to(DEVICE)

                outputs = model(images, input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= total_val
        val_acc = correct_val / total_val
        
        duration = time.time() - start_time
        print(f"Epoch {epoch+1}/{EPOCHS} [{duration:.0f}s] - Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_multimodal_model.pth')
            print("--> Đã lưu mô hình tốt nhất!")

    # Final Evaluation
    print("\n--- Đánh giá chi tiết ---")
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    print(f"Accuracy: {best_val_acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

if __name__ == "__main__":
    train_model()
