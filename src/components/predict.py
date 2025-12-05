import pickle
import pandas as pd
#from src.utils import text_cleaning


def load_artifacts():
    with open('artifacts/vectorizer.pkl','rb')as f:
        vectorizer=pickle.load(f)
    with open('artifacts/best_model.pkl','rb')as f:
        model=pickle.load(f)
    return vectorizer,model    

def prediction(text):
    #cleaned_text=text_cleaning(text)
    vectorizer,model=load_artifacts()
    vectorized_text=vectorizer.transform(text)
    predicted=model.predict(vectorized_text)

    if(predicted[0]==0):
        return "FAKE NEWS"
    else:
        return "REAL NEWS"