"""
Train and export the TruthGuard models using the same workflow as the original Colab notebook.

Expected CSV files:
    data/Fake.csv
    data/True (1).csv

Run:
    python train_and_save.py
"""
from pathlib import Path
import pickle
import re
import string
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

def wordopt(text):
    text = str(text).lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\n", "", text)
    text = re.sub(r"\w*\d\w*", "", text)
    return text

fake_path = DATA_DIR / "Fake.csv"
true_path = DATA_DIR / "True (1).csv"
if not fake_path.exists() or not true_path.exists():
    raise FileNotFoundError(
        "Dataset files not found. Put Fake.csv and True (1).csv inside the data/ folder."
    )

data_fake = pd.read_csv(fake_path)
data_true = pd.read_csv(true_path)
data_fake["class"] = 0
data_true["class"] = 1

# Keep the original notebook's 10-record holdout from each class.
data_fake = data_fake.iloc[:-10]
data_true = data_true.iloc[:-10]
data_merge = pd.concat([data_fake, data_true], axis=0)
data = data_merge.drop(["title", "subject", "date"], axis=1)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

data["text"] = data["text"].astype(str).apply(wordopt)
x = data["text"]
y = data["class"]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42
)

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

models = {
    "logistic_regression.pkl": LogisticRegression(),
    "decision_tree.pkl": DecisionTreeClassifier(),
    "gradient_boosting.pkl": GradientBoostingClassifier(random_state=0),
    "random_forest.pkl": RandomForestClassifier(random_state=0),
}
for filename, model in models.items():
    model.fit(xv_train, y_train)
    score = model.score(xv_test, y_test)
    print(f"{filename}: test accuracy = {score:.4f}")
    with (MODEL_DIR / filename).open("wb") as f:
        pickle.dump(model, f)

with (MODEL_DIR / "vectorizer.pkl").open("wb") as f:
    pickle.dump(vectorization, f)

print(f"\nExported vectorizer and {len(models)} models to: {MODEL_DIR}")
