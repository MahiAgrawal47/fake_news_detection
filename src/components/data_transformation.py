import pickle
from src.utils import get_vectorizer, fit_vectorizer, transform_vectorizer
import pandas as pd

print("🚀 data_transformation.py STARTED")

class DataTransformation:
    def transform(self):
        
        train_df = pd.read_csv('artifacts/train.csv')
       
        test_df = pd.read_csv('artifacts/test.csv')
        

        
        #train_df['text']=train_df['text'].apply(text_cleaning)
        
       # test_df['text']=test_df['text'].apply(text_cleaning)
        
        vectorizer = get_vectorizer()
        
        # Fit only on train
        x_train = fit_vectorizer(vectorizer, train_df['text'])
        
        # Transform only on test
        x_test = transform_vectorizer(vectorizer, test_df['text'])

        pickle.dump(vectorizer, open("artifacts/vectorizer.pkl", "wb"))

        
        y_train=train_df['output']
        y_test=test_df['output']
        return x_train, y_train, x_test, y_test
    

if __name__ == "__main__":
    obj = DataTransformation()
    print("Calling transform()...")  # <-- Add this
    x_train, y_train, x_test, y_test= obj.transform()
    print("Transform Completed.")

