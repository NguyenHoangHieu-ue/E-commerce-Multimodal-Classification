# Project: Shopee Product Matching - Data Mining Course

## 🎯 Core Objectives
1. **Multimodal Pipeline:** Build and maintain a reproducible pipeline combining **Image (ResNet50)** and **Text (BERT)** features.
2. **Efficiency:** Optimize training for CPU/GPU environments using subset strategies where necessary.
3. **User-Friendly Demo:** Provide a web-based interface (Streamlit) for real-time inference comparison.

## 📂 Project Structure & Key Files
- `Shopee_Multimodal_Project.ipynb`: The main research notebook containing EDA, training, and evaluation.
- `src/expand_silver.py`: Primary tool for dataset expansion (Silver Dataset) using keyword heuristics.
- `app.py`: Streamlit application for the web demo (Root entry point).
- `docs/silver_dataset.csv`: Training data (5,000 samples, auto-labeled).
- `docs/gold_dataset_labeled.csv`: Evaluation data (300 samples, human-labeled).

## 🤖 AI Agent Mandates

### 1. Model Versions (Strict)
Only support the following two tiers. Do NOT re-introduce MobileNet (V2):
- **Baseline V1:** SimpleCNN (Images) + LSTM (Text).
- **Advanced V3:** Multimodal Fusion (ResNet50 + BERT via Concatenation).

### 2. Dataset Management
- **Raw Data:** Never modify files in `shopee-product-matching/`.
- **Expansion Logic:** To update categories or keywords, edit the `QUY_TAC_TU_KHOA` dictionary in `src/expand_silver.py` and run it to overwrite `docs/silver_dataset.csv`.
- **Git Sync:** Always allow `docs/silver_dataset.csv` in `.gitignore` as it is required for the notebook to function immediately after cloning.

### 3. Training & Evaluation Workflows
- **CPU Constraints:** If training on CPU, use the `SAMPLE_SIZE` strategy (see `src/quick_eval.py`) to provide feedback without stalling.
- **Metrics:** Always report **Accuracy, Precision, Recall, and F1-Score** (Weighted).
- **Visualization:** Every significant change must be validated by running the Confusion Matrix visualization at the end of the notebook.

### 4. Technical Stack
- **Frameworks:** PyTorch, Transformers (HuggingFace), Streamlit.
- **Backbones:** `resnet50` (Visual), `bert-base-multilingual-cased` (Text).

## 📝 Communication Style
- All code comments and UI labels in specialized scripts (`src/`) and the Demo (`app_demo.py`) should be in **Vietnamese**.
- Professional, concise technical rationale is preferred.

---
*Last Updated: 2026-05-23 by Gemini CLI (Cleanup & Expansion Phase)*
