import pickle
import re
import string
from pathlib import Path

import pandas as pd

MODEL_FILES = {
    "LR": "logistic_regression.pkl",
    "DT": "decision_tree.pkl",
    "GB": "gradient_boosting.pkl",
    "RF": "random_forest.pkl",
}

def wordopt(text: str) -> str:
    """Keep preprocessing consistent with the user's original Colab notebook."""
    text = text.lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\n", "", text)
    text = re.sub(r"\w*\d\w*", "", text)
    return text

def load_artifacts(model_dir: Path):
    vectorizer_path = model_dir / "vectorizer.pkl"
    if not vectorizer_path.exists():
        raise FileNotFoundError(
            "Saved model files are missing. Run train_and_save.py first after placing "
            "Fake.csv and True (1).csv in the project data/ folder."
        )
    with vectorizer_path.open("rb") as f:
        vectorizer = pickle.load(f)

    models = {}
    missing = []
    for key, filename in MODEL_FILES.items():
        path = model_dir / filename
        if not path.exists():
            missing.append(filename)
            continue
        with path.open("rb") as f:
            models[key] = pickle.load(f)
    if missing:
        raise FileNotFoundError("Missing model artifact(s): " + ", ".join(missing))
    return {"vectorizer": vectorizer, "models": models}

def analyze_news(news: str, artifacts: dict):
    cleaned = wordopt(news)
    features = artifacts["vectorizer"].transform([cleaned])
    model_labels = {"LR": "Logistic Regression", "DT": "Decision Tree",
                    "GB": "Gradient Boosting", "RF": "Random Forest"}
    output = {}
    for key, model in artifacts["models"].items():
        prediction = int(model.predict(features)[0])
        output[key] = {
            "class": prediction,
            "label": "Real News" if prediction == 1 else "Fake News",
            "model_name": model_labels[key],
        }
    return {"models": output}
