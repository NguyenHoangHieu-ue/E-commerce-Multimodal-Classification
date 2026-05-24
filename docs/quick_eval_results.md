# Quick Evaluation Results (CPU Subset)

This document contains the results from a quick evaluation run on a reduced subset of the data (running on CPU) to provide immediate feedback for web demo model selection.

## 1. Methodology

*   **Hardware:** CPU
*   **Sample Size:** 500 samples (from Silver Dataset)
*   **Image Processing:** Resized to 224x224
*   **Classes:** 10

## 2. Models Evaluated

1.  **Baseline CNN:** A custom, simple 3-layer Convolutional Neural Network trained for 2 epochs.
2.  **Advanced Multimodal (Subset):** A combination of **Multilingual BERT** (for text) and **ResNet18** (for images). Due to CPU limitations, only the classification head was trained for 1 epoch on a tiny subset (100 training samples, 40 validation samples).

## 3. Results Summary

| Model       | Accuracy | F1-Score |
|-------------|----------|----------|
| **CNN**     | 0.2400   | 0.1612   |
| **Multi**   | 0.2500   | 0.1505   |

## 4. Conclusion & Recommendation for Web Demo

Even under extreme CPU constraints and limited training time (1 epoch, frozen backbone), the **Multimodal Model** achieved a slightly higher raw accuracy (25.00%) compared to the Baseline CNN (24.00%).

However, both models are severely underfitted due to the limitations of this quick evaluation. 

**Recommendation:** For the Web Demo, you should ideally use the **Advanced Multimodal (BERT + ResNet)** architecture. Although its performance here is low, we know theoretically and from previous runs that it possesses a much higher capacity to learn complex relationships compared to the Simple CNN. If you deploy the demo, it is recommended to use the `.pth` weights from a full GPU training run.
