from flask import Flask, render_template, request
import pickle
from src.utils import text_cleaning

app = Flask(__name__)

vectorizer = pickle.load(open("artifacts/vectorizer.pkl", "rb"))
model = pickle.load(open("artifacts/best_model.pkl", "rb"))

@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["news_text"]
    cleaned = text_cleaning(text)
    vector = vectorizer.transform([cleaned])
    pred = model.predict(vector)[0]

    result = "REAL NEWS ✔" if pred == 1 else "FAKE NEWS ❌"
    result_class = "real" if pred == 1 else "fake"

    return render_template("home.html", result=result, result_class=result_class)

if __name__ == "__main__":
    app.run(debug=True)
