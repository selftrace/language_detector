data["Text"] = data["Text"].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    data["Text"],
    data["Language"],
    test_size=0.2,
    random_state=42
)

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2,4)
)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier(random_state=42)
}


accuracies = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = model.score(X_test, y_test)

    accuracies[name] = accuracy

    print(f"\n{name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, predictions))

plt.figure(figsize=(8,5))

plt.bar(
    accuracies.keys(),
    accuracies.values()
)

plt.ylabel("Accuracy")
plt.title("Model Comparison")

plt.xticks(rotation=15)

plt.tight_layout()
plt.show()

def detect_language(text):

    cleaned = clean_text(text)

    transformed = vectorizer.transform([cleaned])

    print("\nPredictions\n")

    for name, model in models.items():

        prediction = model.predict(transformed)[0]

        print(f"{name}: {prediction}")


while True:

    sentence = input("\nEnter text (or type quit): ")

    if sentence.lower() == "quit":
        break

    detect_language(sentence)