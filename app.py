from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
from lime.lime_text import LimeTextExplainer

app = Flask(__name__)

explainer = LimeTextExplainer(class_names=["Correct", "Incorrect"])

def preprocess(text):
    text = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)
    text = re.sub(r'[أإآ]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ـ+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_proba(texts):
    texts = [preprocess(t) for t in texts]

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )

    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    return probs.detach().cpu().numpy()

model_path = "my_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

labels = ["استدلال صحيح ✅", "استدلال غير صحيح ❌"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        ayah_raw = data.get("ayah", "").strip()
        claim_raw = data.get("claim", "").strip()

        if not ayah_raw or not claim_raw:
            return jsonify({
                "result": "⚠️ يرجى تعبئة جميع الحقول قبل التحقق"
                })

        ayah = preprocess(ayah_raw)
        claim = preprocess(claim_raw)

        text = ayah + " [SEP] " + claim

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128
        )

        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()

        return jsonify({
            "result": labels[prediction]
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"result": "حدث خطأ أثناء التحقق"})

@app.route("/explain", methods=["POST"])
def explain():
    data = request.get_json()

    ayah = preprocess(data["ayah"])
    claim = preprocess(data["claim"])

    text = "ayah: " + ayah + " [SEP] tafsir: " + claim
    exp = explainer.explain_instance(text, predict_proba, num_features=5)

    return jsonify(exp.as_list())

if __name__ == "__main__":
    app.run(debug=True, port=5006)