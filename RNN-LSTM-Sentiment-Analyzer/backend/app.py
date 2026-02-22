from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask_cors import CORS
import re

app = Flask(__name__)
CORS(app)

VOCAB_SIZE = 10000
MAX_LEN = 200
INDEX_OFFSET = 3
UNKNOWN_TOKEN = 2

rnn_model = load_model("../models/rnn_model.h5")
lstm_model = load_model("../models/lstm_model.h5")

word_index = imdb.get_word_index()


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)

    words = text.split()
    sequence = []

    for word in words:
        idx = word_index.get(word, UNKNOWN_TOKEN) + INDEX_OFFSET
        if idx >= VOCAB_SIZE:
            idx = UNKNOWN_TOKEN
        sequence.append(idx)

    return pad_sequences([sequence], maxlen=MAX_LEN)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json() or {}

    text = (
        data.get("text")
        or data.get("review")
        or data.get("input")
        or ""
    ).strip()

    if not text:
        return jsonify({"error": "Text is required"}), 400

    seq = preprocess_text(text)

    rnn_score = float(rnn_model.predict(seq, verbose=0)[0][0])
    lstm_score = float(lstm_model.predict(seq, verbose=0)[0][0])

    return jsonify({
        "rnn_result": {
            "score": round(rnn_score, 2),
            "label": "Positive" if rnn_score >= 0.5 else "Negative"
        },
        "lstm_result": {
            "score": round(lstm_score, 2),
            "label": "Positive" if lstm_score >= 0.5 else "Negative"
        }
    })
if __name__ == "__main__":
    app.run(debug=True)