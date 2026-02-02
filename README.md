# 🪶 LakotaBERT: Low-Resource Language Model

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Model: RoBERTa](https://img.shields.io/badge/Model-RoBERTa-blue)](https://huggingface.co/docs/transformers/model_doc/roberta)
[![Language: Lakota](https://img.shields.io/badge/Language-Lakota%20(lkt)-green)](https://en.wikipedia.org/wiki/Lakota_language)

**📅 Project Date:** Spring 2024
**🧠 Model Architecture:** RoBERTa (Robustly Optimized BERT)
**📉 Objective:** Masked Language Modeling (MLM) for Endangered Language Revitalization
**🛠️ Tech Stack:** PyTorch, Hugging Face Transformers, Tesseract OCR, Python

---

### 📖 Research Abstract
Lakota is a critically endangered Siouan language with limited digital resources. This project introduces **LakotaBERT**, the first Large Language Model (LLM) tailored specifically for Lakota.

Unlike English-based models that fail to capture the agglutinative morphology of Native American languages, LakotaBERT was pre-trained from scratch on a custom-compiled corpus. The model achieved a **Masked Language Modeling (MLM) accuracy of ~61%**, demonstrating the viability of Transformer models for language revitalization.

---

### 📊 Performance Metrics
We evaluated the model on a held-out validation set. The detailed results are below:

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Accuracy** | **61.15%** | Percentage of masked tokens correctly predicted |
| **Precision** | **0.6150** | High precision in token retrieval |
| **F1 Score** | **0.6025** | Harmonic mean of precision and recall |
| **MRR** | **0.6115** | Mean Reciprocal Rank (Rank accuracy of the correct token) |
| **CER** | **0.2913** | Character Error Rate (Lower is better) |

---

### 🏗️ Pipeline Architecture
The project followed a standard NLP research pipeline:

1.  **Data Acquisition:** Aggregated 105k sentences from bilingual dictionaries, oral histories, and websites. Used **Tesseract OCR** to digitize physical texts.
2.  **Tokenization:** Trained a Byte-Pair Encoding (BPE) tokenizer (Vocab Size: 52k) to handle Lakota's complex suffixes and prefixes.
3.  **Pre-training:** Utilized the **RoBERTa** architecture with a 15% dynamic masking strategy.

---

### 💻 Implementation Details
The training script utilizes the Hugging Face `Trainer` API with optimized hyperparameters for low-resource settings:

```python
# Configuration for Low-Resource setting (from src/train_lakota_roberta.py)
config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,  # Optimized for smaller dataset size
    type_vocab_size=1,
)
