import 
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

# we load and clean
DATA_PATH = "Language Detection.csv"  

data = (
    pd.read_csv(DATA_PATH)
    [["Text", "Language"]]
    .dropna()
    .drop_duplicates()
)

def clean(text):
    text = str(text).lower()
    text = re.sub(r"[_\d]+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()

X_train, X_test, y_train, y_test = train_test_split(
    data["Text"],
    data["Language"],
    test_size=0.2,
    random_state=42,
    stratify=data["Language"]
)

print(f"Samples: {len(data):,}")
print(f"Languages: {data['Language'].nunique()}")
print("\nClass distribution:")
print(data["Language"].value_counts())

# NLP
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        preprocessor=clean,
        analyzer="char",
        sublinear_tf=True
    )),
    ("model", LinearSVC(class_weight="balanced"))
])

# tune both text representation and classifier
params = {
    "tfidf__ngram_range": [(2, 4), (2, 5), (3, 5)],
    "tfidf__min_df": [1, 2],
    "model__C": [0.5, 1, 2]
}

search = GridSearchCV(
    pipeline,
    params,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1
)

search.fit(X_train, y_train)

model = search.best_estimator_
pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)
macro_f1 = f1_score(y_test, pred, average="macro")

print("\nBest parameters:", search.best_params_)
print(f"Cross-validation F1: {search.best_score_:.4f}")
print(f"Test accuracy:       {accuracy:.4f}")
print(f"Test macro F1:       {macro_f1:.4f}")
print("\n", classification_report(y_test, pred))

#confusion matrix
labels = sorted(y_test.unique())

fig, ax = plt.subplots(figsize=(10, 10))
ConfusionMatrixDisplay.from_predictions(
    y_test,
    pred,
    labels=labels,
    xticks_rotation=90,
    ax=ax,
    colorbar=False
)
plt.title("Language Detection — Confusion Matrix")
plt.tight_layout()
plt.show()

#error analysis
results = pd.DataFrame({
    "text": X_test.reset_index(drop=True),
    "actual": y_test.reset_index(drop=True),
    "predicted": pred
})

results["length"] = results["text"].str.len()
results["correct"] = results["actual"] == results["predicted"]
results["length_group"] = pd.cut(
    results["length"],
    [0, 10, 25, 50, 100, np.inf],
    labels=["1–10", "11–25", "26–50", "51–100", "100+"]
)

print("\nAccuracy by text length:")
print(
    results.groupby("length_group", observed=False)["correct"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "accuracy"})
)

errors = results[~results["correct"]]

print(f"\nMisclassified samples: {len(errors)}")
print(
    errors[["text", "actual", "predicted"]]
    .head(15)
    .to_string(index=False)
)

#commong language confusions
print("\nMost common confusions:")
print(
    errors.groupby(["actual", "predicted"])
    .size()
    .sort_values(ascending=False)
    .head(10)
)

#optimized as much possible
joblib.dump(model, "language_detector.pkl")
print("\nModel saved as language_detector.pkl")

while True:
    text = input("\nEnter text ('quit' to exit): ").strip()

    if text.lower() == "quit":
        break

    if not text:
        print("Please enter some text.")
        continue

    print("Predicted language:", model.predict([text])[0])
