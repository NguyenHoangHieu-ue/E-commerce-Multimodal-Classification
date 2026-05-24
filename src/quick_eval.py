import os
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
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModel
import time

# --- CONFIG ---
TRAIN_CSV = 'shopee-product-matching/train.csv'
IMAGE_DIR = 'shopee-product-matching/train_images/'
SILVER_CSV = 'docs/silver_dataset.csv'
BERT_MODEL_NAME = 'bert-base-multilingual-cased'
DEVICE = torch.device('cpu')
SAMPLE_SIZE = 500  # Small sample for quick CPU feedback
BATCH_SIZE = 8

# --- MODELS ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        return self.fc(hidden[-1])

class MultimodalModel(nn.Module):
    def __init__(self, num_classes, freeze_backbone=True):
        super(MultimodalModel, self).__init__()
        resnet = models.resnet18(pretrained=True) # Use ResNet18 for CPU speed
        self.image_branch = nn.Sequential(*list(resnet.children())[:-1])
        self.text_branch = AutoModel.from_pretrained(BERT_MODEL_NAME)
        if freeze_backbone:
            for param in self.image_branch.parameters(): param.requires_grad = False
            for param in self.text_branch.parameters(): param.requires_grad = False
        self.fusion_dim = 512 + 768
        self.classifier = nn.Sequential(nn.Linear(self.fusion_dim, 256), nn.ReLU(), nn.Linear(256, num_classes))
    def forward(self, image, input_ids, attention_mask):
        img_features = self.image_branch(image).view(image.size(0), -1)
        text_outputs = self.text_branch(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]
        combined = torch.cat((img_features, text_features), dim=1)
        return self.classifier(combined)

# --- UTILS ---
class SimpleTokenizer:
    def __init__(self, max_words=1000):
        self.max_words = max_words
        self.word_index = {'<PAD>': 0}
    def fit(self, texts):
        words = " ".join(texts).lower().split()
        from collections import Counter
        common = Counter(words).most_common(self.max_words - 1)
        for i, (w, _) in enumerate(common): self.word_index[w] = i + 1
    def tokenize(self, texts, max_len=64):
        seqs = []
        for t in texts:
            words = t.lower().split()
            seq = [self.word_index.get(w, 0) for w in words][:max_len]
            seq += [0] * (max_len - len(seq))
            seqs.append(seq)
        return torch.tensor(seqs, dtype=torch.long)

class ShopeeDataset(Dataset):
    def __init__(self, df, transform, tokenizer=None, is_multimodal=False):
        self.df = df; self.transform = transform; self.tokenizer = tokenizer; self.is_multimodal = is_multimodal
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self.transform(Image.open(os.path.join(IMAGE_DIR, row['image'])).convert('RGB'))
        label = torch.tensor(row['label'], dtype=torch.long)
        if self.is_multimodal:
            inputs = self.tokenizer(row['title'], padding='max_length', truncation=True, max_length=64, return_tensors='pt')
            return img, inputs['input_ids'].flatten(), inputs['attention_mask'].flatten(), label
        else:
            return img, label

def main():
    print("--- Start Quick Evaluation Pipeline (CPU) ---")
    df = pd.read_csv(SILVER_CSV).sample(SAMPLE_SIZE, random_state=42)
    le = LabelEncoder(); df['label'] = le.fit_transform(df['category'])
    num_classes = len(le.classes_)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    
    # 1. Baseline CNN
    print("\nTraining Baseline CNN...")
    train_ds = ShopeeDataset(train_df, transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ShopeeDataset(val_df, transform), batch_size=BATCH_SIZE)
    
    model_cnn = SimpleCNN(num_classes)
    opt = optim.Adam(model_cnn.parameters(), lr=0.001); crit = nn.CrossEntropyLoss()
    
    for epoch in range(2):
        model_cnn.train()
        for imgs, labels in train_loader:
            opt.zero_grad(); crit(model_cnn(imgs), labels).backward(); opt.step()
    
    model_cnn.eval(); preds, truths = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            preds.extend(torch.argmax(model_cnn(imgs), 1).numpy()); truths.extend(labels.numpy())
    
    acc_cnn = accuracy_score(truths, preds)
    f1_cnn = f1_score(truths, preds, average='weighted')
    print(f"CNN Result - Acc: {acc_cnn:.4f}, F1: {f1_cnn:.4f}")

    # 2. Advanced Multimodal (Very limited due to CPU)
    print("\nEvaluating Advanced Multimodal (Subset)...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    train_ds_multi = ShopeeDataset(train_df.iloc[:100], transform, tokenizer, is_multimodal=True) # Even smaller subset
    train_loader_multi = DataLoader(train_ds_multi, batch_size=4, shuffle=True)
    val_loader_multi = DataLoader(ShopeeDataset(val_df.iloc[:40], transform, tokenizer, is_multimodal=True), batch_size=4)
    
    model_adv = MultimodalModel(num_classes)
    opt_adv = optim.Adam(model_adv.parameters(), lr=2e-5)
    
    # Train 1 epoch on tiny subset just to get a signal
    model_adv.train()
    for imgs, ids, masks, labels in train_loader_multi:
        opt_adv.zero_grad(); crit(model_adv(imgs, ids, masks), labels).backward(); opt_adv.step()
        
    model_adv.eval(); preds, truths = [], []
    with torch.no_grad():
        for imgs, ids, masks, labels in val_loader_multi:
            preds.extend(torch.argmax(model_adv(imgs, ids, masks), 1).numpy()); truths.extend(labels.numpy())
            
    acc_adv = accuracy_score(truths, preds)
    f1_adv = f1_score(truths, preds, average='weighted')
    print(f"Advanced Result - Acc: {acc_adv:.4f}, F1: {f1_adv:.4f}")

    print("\n--- SUMMARY ---")
    print(f"| Model | Accuracy | F1-Score |")
    print(f"|-------|----------|----------|")
    print(f"| CNN   | {acc_cnn:.4f}   | {f1_cnn:.4f}   |")
    print(f"| Multi | {acc_adv:.4f}   | {f1_adv:.4f}   |")

if __name__ == "__main__":
    main()
