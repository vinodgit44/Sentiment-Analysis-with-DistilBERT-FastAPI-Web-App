from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, pipeline
import os

MODEL_DIR = "./results"

# -----------------------------
# LOAD CONFIG + FIX LABEL NAMES
# -----------------------------
config = AutoConfig.from_pretrained(MODEL_DIR)

if not config.id2label or "LABEL" in list(config.id2label.values())[0]:
    num_labels = config.num_labels

    if num_labels == 2:
        config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
        config.label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    elif num_labels == 3:
        config.id2label = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
        config.label2id = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}

    config.save_pretrained(MODEL_DIR)

# -----------------------------
# LOAD TOKENIZER & MODEL
# -----------------------------
if not os.path.exists(os.path.join(MODEL_DIR, "tokenizer.json")):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokenizer.save_pretrained(MODEL_DIR)
else:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, config=config)

pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# -----------------------------
# FASTAPI APP
# -----------------------------
app = FastAPI()

@app.get("/")
def home():
    return {"message": "Sentiment model is running!"}


# -----------------------------
# BOOTSTRAP UI PAGE
# -----------------------------
@app.get("/ui", response_class=HTMLResponse)
def ui_page():
    html = """
    <html>
    <head>
        <title>Sentiment App</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>

    <body class="bg-light">

        <div class="container py-5">
            <div class="row justify-content-center">
                <div class="col-md-8">

                    <div class="card shadow-lg p-4">
                        <h2 class="text-center mb-4">Sentiment Analyzer 🙂</h2>

                        <form action="/predict_ui" method="post">
                            <div class="mb-3">
                                <label class="form-label"><b>Enter text:</b></label>
                                <textarea name="text" class="form-control" rows="4"
                                    placeholder="Type something..."></textarea>
                            </div>

                            <div class="text-center">
                                <button class="btn btn-primary btn-lg px-4" type="submit">
                                    Analyze Sentiment
                                </button>
                            </div>
                        </form>

                    </div>

                </div>
            </div>
        </div>

    </body>
    </html>
    """
    return HTMLResponse(html)


# -----------------------------
# UI PREDICTION HANDLER
# -----------------------------
@app.post("/predict_ui", response_class=HTMLResponse)
async def predict_ui(text: str = Form(...)):
    result = pipe(text)[0]
    label = result["label"]
    score = round(result["score"], 4)

    if label.upper() == "POSITIVE":
        emoji, color = "🙂", "success"
    elif label.upper() == "NEGATIVE":
        emoji, color = "😡", "danger"
    else:
        emoji, color = "😐", "warning"

    html = f"""
    <html>
    <head>
        <title>Sentiment Result</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>

    <body class="bg-light">

        <div class="container py-5">
            <div class="row justify-content-center">
                <div class="col-md-8">

                    <div class="card shadow-lg p-4">

                        <h2 class="text-center mb-4">Sentiment Result</h2>

                        <div class="alert alert-{color} text-center" style="font-size:22px;">
                            <b>{label} {emoji}</b><br>
                            Confidence: {score}
                        </div>

                        <p><b>Input:</b> {text}</p>

                        <div class="text-center mt-3">
                            <a href="/ui" class="btn btn-secondary btn-lg">Try Another</a>
                        </div>

                    </div>

                </div>
            </div>
        </div>

    </body>
    </html>
    """
    return HTMLResponse(html)


# -----------------------------
# RAW API
# -----------------------------
@app.post("/predict")
def predict(text: str):
    return pipe(text)
