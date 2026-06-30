# Fake News Detector

A machine learning web app that classifies news text as REAL or FAKE, built with scikit-learn and Flask.

## Overview

This project trains a text classification model (TF-IDF + Logistic Regression / Naive Bayes / LinearSVC, with the best performer selected automatically) to predict whether a piece of news text is real or fake, and serves predictions through a simple Flask web interface.

## Tech Stack

- **Python** – core language
- **scikit-learn** – TF-IDF vectorization, model training, evaluation
- **NLTK** – text preprocessing (stopword removal, lemmatization)
- **Flask** – web app for serving predictions
- **pandas** – data handling

## Project Structure

```
Fake_news_detection/
├── data/
│   └── combined_news.csv
├── artifacts/
│   ├── train.csv
│   ├── test.csv
│   ├── vectorizer.pkl
│   └── best_model.pkl
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_train.py
│   ├── predict.py
│   └── utils.py
├── templates/
│   └── home.html
├── static/
│   └── styles.css
├── app.py
└── README.md
```

## How It Works

1. **Data Ingestion** – splits the dataset into train/test sets
2. **Data Transformation** – cleans text (removes source artifacts, stopwords, lemmatizes) and converts it to TF-IDF vectors
3. **Model Training** – trains multiple models (Logistic Regression, Naive Bayes, random forest), evaluates each, and saves the best-performing one
4. **Prediction** – the Flask app loads the saved vectorizer and model to classify user-submitted text in real time

## Setup

## Installation

```bash
git clone <repo-url>
cd Fake_news_detection
pip install -r requirements.txt
```

## Run the Application

```bash
python app.py
```

Then open the URL displayed in the terminal.

## Retrain the Model (Optional)

If you want to train the model from scratch:

```bash
python -m src.components.data_ingestion
python -m src.components.data_transformation
python -m src.components.model_train
```

After training, launch the application:

```bash
python app.py
```


## Future Improvements

- Add cross-validation and a more diverse, multi-source dataset for better generalization
- Experiment with transformer-based models (e.g. DistilBERT) for richer text representations
- Add a confidence score alongside the prediction
- Deploy the app (e.g. Render, Railway, or Hugging Face Spaces)

