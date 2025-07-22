# 📚 Abstractive Summarization of Educational Video Transcripts Using Transformer Models

_A project by Harshil Pradhan_

---

## 📖 Overview

This project focuses on **summarizing video lecture transcriptions** using state-of-the-art deep learning models. The objective is to transform long, often noisy transcripts into concise, structured summaries. We evaluate and compare the performance of three leading transformer-based models—**T5**, **Pegasus**, and **BART**—and introduce an **Ensemble model** to enhance the quality and robustness of summaries.

The **VT-SSum dataset**, purpose-built for video transcript segmentation and summarization, is used to fine-tune these models. This effort makes educational content more consumable, searchable, and accessible.

---

## 🛠️ Methodology

### 📦 Dataset Preparation

- **Dataset:** VT-SSum
    - 9,616 videos
    - 125,000 transcript-summary pairs
- Transcripts are cleaned, segmented, and tokenized into input-output pairs for summarization.

### 🤖 Model Selection

- **T5:** Text-to-Text Transfer Transformer by Google
- **Pegasus:** Optimized summarization model by Google
- **BART:** Bidirectional Auto-Regressive Transformer by Facebook AI
- **Ensemble Model** (Proposed): Combines predictions from all three models via averaging or voting techniques.

### ⚙️ Training & Evaluation

- Models were fine-tuned using GPU acceleration.
- Dataset was split into **80% training** and **20% validation**.
- **Early stopping** was applied to avoid overfitting.
- ROUGE metrics were used for performance evaluation.

### 🔁 Methodology Flowchart

![Methodology Flowchart](https://github.com/user-attachments/assets/2eb1efe8-10e0-44df-89ec-49cffc420404)

---

## 🏗️ Model Architectures

### 🔷 T5 (Text-to-Text Transfer Transformer)

- Developed by Google.
- Converts all tasks into a unified text-to-text form.
- Uses an encoder-decoder architecture that performs exceptionally well in summarization tasks.

### 🔷 Pegasus

- Also by Google, designed specifically for abstractive summarization.
- Pre-training involves masking and predicting entire sentences, enabling excellent understanding of salient content.
- Delivers high-quality, coherent summaries.

### 🔷 BART (Bidirectional and Auto-Regressive Transformers)

- Developed by Facebook AI.
- Trained as a denoising autoencoder: corrupts inputs and learns to recover the original.
- Blends BERT-style encoding with GPT-style decoding for strong generative performance.

### 🔷 Ensemble Model (Proposed)

- Combines the predictions of T5, Pegasus, and BART using voting or averaging strategies.
- Aims to produce more consistent and accurate summaries by integrating the strengths of each individual model.
- **Trained on a single P100 GPU**, achieving improved ROUGE scores.

---

## 🖼️ Architecture Images

### 💡 BART Architecture

![BART Architecture](https://github.com/user-attachments/assets/e153c796-0ce5-4992-850c-1d41f955ce8d)

### 💡 Pegasus Architecture

![Pegasus Architecture](https://github.com/user-attachments/assets/3338fe48-eade-4b0c-9667-dce9b9fd181f)

### 💡 T5 Architecture

![T5 Architecture](https://github.com/user-attachments/assets/b88920b3-9fa1-4b5c-a675-cdf0ea7ee5f0)

---

## 📊 Results & Performance

| Model            | ROUGE-1 | ROUGE-2 | ROUGE-L | Execution Time (hrs) | GPU Used     |
|------------------|---------|---------|---------|----------------------|--------------|
| T5               | 0.90    | 0.83    | 0.88    | 10–12                | NVIDIA T4 ×2 |
| Pegasus          | 0.87    | 0.81    | 0.85    | 8–10                 | NVIDIA T4 ×2 |
| BART             | 0.86    | 0.80    | 0.84    | 6–8                  | NVIDIA T4 ×2 |
| 🌟 **Ensemble**  | 0.91    | 0.84    | 0.89    | 12                   | NVIDIA P100  |

- **T5** achieved the best performance among the standalone models.
- The **Ensemble Model** produced the highest overall ROUGE scores and more stable results across different input types.
- Ensemble utilized a **single NVIDIA P100 GPU** and completed in approximately **12 hours**.

---

## 🚀 Future Work

- 🔍 Explore additional transformer models such as LongT5, LED, or GPT variants.
- 🎯 Perform automated hyperparameter tuning using frameworks like Optuna or Ray.
- 🧠 Apply data augmentation techniques (paraphrasing, back-translation).
- 📹 Integrate multimodal data such as video frames or audio transcripts.
- 🌐 Deploy as a web service or plugin for use in e-learning platforms (e.g., Moodle, Coursera).

---

## 📜 References

1. T. Lv, L. Cui, M. Vasilijevic, and F. Wei, "[VT-SSum: A Benchmark Dataset for Video Transcript Segmentation and Summarization](https://arxiv.org/abs/2107.13485)," arXiv, 2021.
2. A. Vaswani et al., "Attention is All You Need," *NIPS*, 2017.
3. M. Lewis et al., "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation," *ACL*, 2020.
4. J. Zhang et al., "PEGASUS: Pre-training with Extracted Gap-Sentences," *ICML*, 2020.

---

## 🤝 Contributions

If you'd like to contribute, open issues or pull requests are welcome!

---

🧑‍💻 Developed by **Harshil Pradhan**
