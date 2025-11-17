# 📘 Sentiment Analysis with DistilBERT  
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange?logo=huggingface)]()
[![Kaggle](https://img.shields.io/badge/Kaggle-GPU%20Accelerated-blue?logo=kaggle)]()
[![Python](https://img.shields.io/badge/Python-3.10-green?logo=python)]()
[![Model](https://img.shields.io/badge/Model-DistilBERT-yellow)]()

📘 Sentiment Analysis with DistilBERT — FastAPI Web App

This project demonstrates a complete end-to-end NLP pipeline using HuggingFace Transformers, DistilBERT, and FastAPI.
It includes:
✔ Dataset loading
✔ Tokenization
✔ Fine-tuning DistilBERT
✔ Saving the model
✔ Building a modern Bootstrap UI
✔ Deploying an API for real-time sentiment prediction


## 🧠 Model: DistilBERT
DistilBERT is 40% smaller, 60% faster, and retains 97% of BERT’s accuracy.  
Ideal for learning NLP and running on free GPUs.

## 🚀 Features
- Fine‑tune DistilBERT in 3–5 minutes on Kaggle GPU  
- HuggingFace `datasets` + `transformers`  
- Mixed precision FP16 training  
- Clean inference pipeline  
- Optional FastAPI deployment  
- Beginner‑friendly explanations

## 📦 Installation

1️⃣ Clone the repo
```bash
git clone https://github.com/yourusername/DistilBERT_Sentiment_Repo.git
cd DistilBERT_Sentiment_Repo
```
📁 Project Structure
```
repo/
│── README.md
│── requirements.txt
│── app.py
│── src/
│   └── inference_example.py
│── notebook_train/
│   └── main.py
```


2️⃣ Create virtual env
```bash
python3 -m venv .venv
source .venv/bin/activate
```


3️⃣ Install dependencies

```python
pip install -r requirements.txt
```

---
## 🏋️ Training (Kaggle Recommended)
Use the Kaggle notebook for:
- GPU acceleration  
- FP16 mixed precision  
- Fast dataset loading
- Training the Model

Your dataset should look like:
train.csv / test.csv
text,label
"I love this product!",1
"This is terrible.",0

Run training:
notebook-train/main.py
```
│── notebook_train/
│   └── main.py
```

What happens:
Tokenizer loads
Dataset is tokenized
DistilBERT is fine-tuned
Metrics (Accuracy, F1) are computed
Model is saved into ./results/

## 📈 Model Performance

After training you will see:
Epoch 1/2 – Accuracy: 0.89, F1: 0.88
Epoch 2/2 – Accuracy: 0.92, F1: 0.91



🛠 Customization
You can modify:
Learning rate
Batch size
Number of labels
Model architecture

Or replace DistilBERT with:
BERT-base
RoBERTa
DeBERTa
ALBERT

## 🔍 Inference
```python
from transformers import pipeline

pipe = pipeline("sentiment-analysis", model="./results")
print(pipe("This movie was great!"))
```

## 🌐 FastAPI Deployment
Run:
```bash
uvicorn app:app --reload
```


🎨 UI
http://127.0.0.1:8000/ui

🧪 API (JSON)
http://127.0.0.1:8000/docs

✨ UI Preview (Description)
Input box for text
Bootstrap card layout
Color-coded results:
Green = Positive 🙂
Red = Negative 😡
Orange = Neutral 😐

🧪 Example Predictions
Text	Output
“I love this!”	POSITIVE 🙂
“This is the worst.”	NEGATIVE 😡
“It works.”	NEUTRAL 😐


📤 Additional Deployment Options
You can deploy this app on:

🔹 HuggingFace Spaces (Free)

Supports Gradio & FastAPI

🔹 AWS EC2

Production + scaling

🔹 Docker
```bash
docker build -t sentiment-app .
docker run -p 8000:8000 sentiment-app
```

## ❤️ Credits
Built using:
HuggingFace Transformers
FastAPI
Bootstrap
PyTorch

## ⭐ Contribute

Pull requests welcome!
You can:
Improve UI
Add datasets
Add multi-language support
Add ONNX optimization



📝 License
MIT License
