# 📘 Sentiment Analysis with DistilBERT  
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange?logo=huggingface)]()
[![Kaggle](https://img.shields.io/badge/Kaggle-GPU%20Accelerated-blue?logo=kaggle)]()
[![Python](https://img.shields.io/badge/Python-3.10-green?logo=python)]()
[![Model](https://img.shields.io/badge/Model-DistilBERT-yellow)]()

A beginner‑friendly, end‑to‑end NLP project that fine‑tunes **DistilBERT** for sentiment analysis using the **IMDB dataset**, **Kaggle GPU accelerators**, and **Hugging Face Transformers**.

---

## 🚀 Features
- Fine‑tune DistilBERT in 3–5 minutes on Kaggle GPU  
- HuggingFace `datasets` + `transformers`  
- Mixed precision FP16 training  
- Clean inference pipeline  
- Optional FastAPI deployment  
- Beginner‑friendly explanations

---

## 📁 Project Structure
```
repo/
│── README.md
│── requirements.txt
│── app.py
│── src/
│   └── inference_example.py
│── notebook/
│   └── training_notebook_placeholder.txt
```

---

## 🧠 Model: DistilBERT
DistilBERT is 40% smaller, 60% faster, and retains 97% of BERT’s accuracy.  
Ideal for learning NLP and running on free GPUs.

---

## 🛠 Installation
```bash
pip install -r requirements.txt
```

---

## 🏋️ Training (Kaggle Recommended)
Use the Kaggle notebook for:
- GPU acceleration  
- FP16 mixed precision  
- Fast dataset loading  

---

## 🔍 Inference
```python
from transformers import pipeline

pipe = pipeline("sentiment-analysis", model="./results")
print(pipe("This movie was great!"))
```

---

## 🌐 FastAPI Deployment
Run:
```bash
uvicorn app:app --reload
```

---

## 📝 License
MIT License
