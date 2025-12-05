import pandas as pd
import os
from sklearn.model_selection import train_test_split

class DataIngestion:
    def initiate_dataingestion(self):
        df=pd.read_csv('data/combined_news.csv')
        

        train,test=train_test_split(df,test_size=0.2,random_state=42)

        os.makedirs('artifacts',exist_ok=True)
        train.to_csv('artifacts/train.csv',index=False,header=True)
        test.to_csv('artifacts/test.csv',index=False,header=True)

        train_path='artifacts/train.csv'
        test_path='artifacts/test.csv'

        return train_path,test_path


if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_dataingestion()
