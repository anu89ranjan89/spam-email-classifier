import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report
# Load dataset
df = pd.read_csv("spam.csv")

# Features and labels
X = df["text"]
y = df["label"]
# TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(X)
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
# Train model
model = MultinomialNB()
model.fit(X_train, y_train)
import pickle

pickle.dump(model, open("spam_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model Saved Successfully!")

# Evaluate
predictions = model.predict(X_test)

from sklearn.metrics import classification_report

print("Accuracy:", accuracy_score(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Test custom messages
while True:
    message = input("\nEnter a message (or type exit): ")

    if message.lower() == "exit":
        break

    transformed = vectorizer.transform([message])

    prediction = model.predict(transformed)[0]

    probabilities = model.predict_proba(transformed)

    spam_prob = probabilities[0][1]

    print("Prediction:", prediction.upper())
    print(f"Spam Probability: {spam_prob*100:.2f}%")