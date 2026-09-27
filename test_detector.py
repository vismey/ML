"""
test_detector.py
Command-line inference script using exclusively the Random Forest Classifier.
"""

import os
import joblib
import numpy as np
import features

MODELS_DIR = "models"


def load_artifacts():
    scaler = joblib.load(os.path.join(MODELS_DIR, "feature_scaler.joblib"))
    pca = joblib.load(os.path.join(MODELS_DIR, "pca_transformer.joblib"))
    rf_model = joblib.load(os.path.join(MODELS_DIR, "random_forest_model.joblib"))
    return scaler, pca, rf_model


def predict_text(text: str):
    scaler, pca, rf_model = load_artifacts()
    feat_dict = features.extract_features_dict(text)
    feat_vector = np.array([[feat_dict[name] for name in features.FEATURE_NAMES]])
    feat_scaled = scaler.transform(feat_vector)

    pred = rf_model.predict(feat_scaled)[0]
    prob = rf_model.predict_proba(feat_scaled)[0]
    pca_coords = pca.transform(feat_scaled)[0].tolist()

    return {
        "model": "Random Forest Ensemble Classifier",
        "prediction": "AI-Generated" if pred == 1 else "Human-Written",
        "confidence": float(prob[pred]),
        "prob_human": float(prob[0]),
        "prob_ai": float(prob[1]),
        "features": feat_dict,
        "pca_coords": pca_coords
    }


if __name__ == "__main__":
    human_sample = "Yesterday my friend and I went to the old diner down the street. The coffee was terrible, but man, the cherry pie made up for it completely!"
    ai_sample = "Artificial intelligence frameworks represent a transformative paradigm in modern computing. Furthermore, systematic optimization protocols ensure scalable performance across enterprise architectures."

    print("\n" + "=" * 55)
    print("Random Forest Text Detector CLI Test")
    print("=" * 55)

    print("\n--- Testing Human Sample ---")
    res_h = predict_text(human_sample)
    print(f"Model: {res_h['model']}")
    print(f"Result: {res_h['prediction']} (Confidence: {res_h['confidence']:.1%})")
    print(f"Probabilities: Human: {res_h['prob_human']:.1%}, AI: {res_h['prob_ai']:.1%}")

    print("\n--- Testing AI Sample ---")
    res_a = predict_text(ai_sample)
    print(f"Model: {res_a['model']}")
    print(f"Result: {res_a['prediction']} (Confidence: {res_a['confidence']:.1%})")
    print(f"Probabilities: Human: {res_a['prob_human']:.1%}, AI: {res_a['prob_ai']:.1%}")
    print("=" * 55)
