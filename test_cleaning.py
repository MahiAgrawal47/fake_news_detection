import pickle

# Load vectorizer and model
vectorizer = pickle.load(open("artifacts/vectorizer.pkl", "rb"))
model = pickle.load(open("artifacts/best_model.pkl", "rb"))

# Test sentences (you can modify these)
test_sentences = [
    "India's GDP grew by 7 percent last quarter.",
    "UPI payments will be banned starting next week.",
    "ISRO successfully launched a new satellite today.",
    "The government announced that all ATM withdrawals will stop tonight.",
]

for text in test_sentences:
    vec = vectorizer.transform([text])  # No cleaning anymore
    pred = model.predict(vec)[0]
    print(f"Text: {text}")
    print(f"Prediction: {pred}")
    print("-" * 40)
