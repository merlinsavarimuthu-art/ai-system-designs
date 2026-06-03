# AI Tokenization Demo using Python

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# -----------------------------
# Step 1: Training Data
# -----------------------------
texts = [
    "I love this product",
    "This is amazing",
    "Absolutely fantastic",
    "I hate this",
    "This is terrible",
    "Worst experience ever"
]

labels = [
    "positive",
    "positive",
    "positive",
    "negative",
    "negative",
    "negative"
]

# -----------------------------
# Step 2: Tokenization
# -----------------------------
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

print("🔹 Tokens created from training data:")
print(vectorizer.get_feature_names_out())

# -----------------------------
# Step 3: Train AI Model
# -----------------------------
model = MultinomialNB()
model.fit(X, labels)

# -----------------------------
# Step 4: Prediction Loop
# -----------------------------
print("\n AI System Ready (type 'exit' to stop)")

while True:
    user_input = input("\nEnter a sentence: ")

    if user_input.lower() == "exit":
        print(" Exiting AI system")
        break

    input_vector = vectorizer.transform([user_input])
    prediction = model.predict(input_vector)

    print("Prediction:", prediction[0])