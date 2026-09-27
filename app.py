"""
app.py
Flask web server for the AI vs Human Text Detector.
Exclusively powered by the Random Forest Ensemble Classifier (Module 2: Learning with Trees).
"""

import os
import json
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

import features

app = Flask(__name__)
CORS(app)

MODELS_DIR = "models"
METRICS_PATH = os.path.join(MODELS_DIR, "metrics_report.json")

# Preload Random Forest artifacts
scaler = None
pca = None
rf_model = None
metrics_data = {}


def load_all_artifacts():
    global scaler, pca, rf_model, metrics_data

    if not os.path.exists(os.path.join(MODELS_DIR, "random_forest_model.joblib")):
        import train
        train.train_and_evaluate()

    scaler = joblib.load(os.path.join(MODELS_DIR, "feature_scaler.joblib"))
    pca = joblib.load(os.path.join(MODELS_DIR, "pca_transformer.joblib"))
    rf_model = joblib.load(os.path.join(MODELS_DIR, "random_forest_model.joblib"))

    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            metrics_data = json.load(f)


# Load artifacts on startup
load_all_artifacts()

SAMPLE_TEXTS = {
    "ai_academic": (
        "Artificial intelligence and machine learning algorithms represent a transformative paradigm in modern data engineering. "
        "Furthermore, modern computational architectures leverage parallel processing frameworks to optimize gradient descent convergence. "
        "In addition, regularized parametric estimators mitigate overfitting by penalizing high-variance weight parameters, "
        "thereby ensuring robust generalization across heterogeneous test distributions."
    ),
    "human_casual": (
        "Honestly, I spent the whole morning trying to fix my bicycle chain and got grease all over my favorite white shirt. "
        "Why do these things always happen right before you need to leave the house? "
        "Anyway, I ended up walking to the metro instead, grabbed a warm chai from the corner vendor, and felt a lot better!"
    ),
    "ai_formal": (
        "Renewable energy systems play a pivotal role in mitigating global carbon emissions and promoting ecological sustainability. "
        "It is worth noting that continuous technological enhancements in photovoltaic cells have significantly increased energy conversion efficiency. "
        "Consequently, policymakers emphasize comprehensive investment in decentralized battery storage to address intermittent grid demands."
    ),
    "human_story": (
        "My grandfather's old pocket watch has been sitting in a wooden drawer since 1998. "
        "Last Sunday, I decided to wind the spring just to see what would happen. "
        "To my astonishment, the tiny brass gears began ticking steadily, as if thirty years had passed in the blink of an eye. "
        "It makes you pause and reflect on how fleeting our modern plastic gadgets really are."
    )
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/samples", methods=["GET"])
def get_samples():
    return jsonify(SAMPLE_TEXTS)


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    return jsonify(metrics_data)


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        text = data.get("text", "").strip()

        if not text:
            return jsonify({"error": "Please provide some text to analyze."}), 400

        # Extract handcrafted features
        feat_dict = features.extract_features_dict(text)
        feat_vector = np.array([[feat_dict[name] for name in features.FEATURE_NAMES]])
        feat_scaled = scaler.transform(feat_vector)

        # Random Forest Prediction & Class Probabilities
        prediction = int(rf_model.predict(feat_scaled)[0])
        probs = rf_model.predict_proba(feat_scaled)[0]
        prob_human = float(probs[0])
        prob_ai = float(probs[1])
        confidence = float(probs[prediction])

        # PCA 2D coordinates for visual cluster projection
        pca_coords = pca.transform(feat_scaled)[0].tolist()

        # Linguistic Explainability Cues
        explanations = []
        burstiness = feat_dict["sentence_length_std"]
        avg_word_len = feat_dict["avg_word_length"]
        discourse_markers = feat_dict["ai_discourse_markers"]
        fk_grade = feat_dict["flesch_kincaid_grade"]
        exclam_q = feat_dict["exclamation_freq"] + feat_dict["question_freq"]

        if burstiness < 3.5:
            explanations.append(f"Low Sentence Burstiness (Std Dev = {burstiness:.1f}): Sentences have uniform lengths, typical of AI generation.")
        else:
            explanations.append(f"High Sentence Burstiness (Std Dev = {burstiness:.1f}): Dynamic mix of short and long sentences, characteristic of natural human rhythm.")

        if discourse_markers > 0.0:
            explanations.append(f"Formal Discourse Markers detected ({discourse_markers:.1f} per 100 words): Matches structured LLM transitional phrasing.")

        if avg_word_len > 5.4:
            explanations.append(f"Elevated Vocabulary Complexity (Avg word length: {avg_word_len:.2f} chars): Higher concentration of multisyllabic academic terms.")
        else:
            explanations.append(f"Conversational Vocabulary (Avg word length: {avg_word_len:.2f} chars): Natural, direct phrasing common in human communication.")

        if exclam_q > 0.15:
            explanations.append(f"Expressive Punctuation (! or ? present): Emotional inflection frequently found in personal human text.")

        if fk_grade > 12.0:
            explanations.append(f"High Readability Grade Level ({fk_grade:.1f}): Complex formal sentence structure.")
        elif fk_grade < 7.0:
            explanations.append(f"Accessible Reading Level ({fk_grade:.1f}): Simple and accessible flow.")

        response = {
            "model": "Random Forest Ensemble Classifier",
            "prediction": "AI-Generated" if prediction == 1 else "Human-Written",
            "prediction_code": prediction,
            "confidence": round(confidence * 100.0, 1),
            "prob_human": round(prob_human * 100.0, 1),
            "prob_ai": round(prob_ai * 100.0, 1),
            "features": feat_dict,
            "pca_coords": [round(float(c), 3) for c in pca_coords],
            "explanations": explanations
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Random Forest AI Detector server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
