from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
#from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.tree import DecisionTreeClassifier
#from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
import pickle
import pandas as pd



class ModelTraining:
    def train_model(self,x_train,y_train,x_test,y_test):

        best_model_name = None
        best_model_obj = None
        best_score = 0

        models={
            "Logistic Regression":LogisticRegression(),
            #"SVC":SVC(),
            "Naive Bayes": MultinomialNB(),
            #"RandomForestClassifier":RandomForestClassifier(),
            #"DecisionTreeClassifier":DecisionTreeClassifier(),
           # "XGBClassifier":XGBClassifier(eval_metric='logloss'),
           "LinearSVC": LinearSVC()
        }
        params={
            "Logistic Regression":{},
            #"SVC":{
             #   'C':[0.1,1,10,100],
              #  'kernel':['linear','rbf','poly'],
               # 'gamma':['scale','auto']
            #},
            "Naive Bayes":{},
            "LinearSVC": {"C": [0.1, 1, 10]}

            # "RandomForestClassifier":{
            #     'n_estimators':[50,100,200],
            #     'max_depth':[None,10,20,30],
            #     'min_samples_split':[2,5,10]
            # },
           
           # "XGBClassifier":{
            #    'n_estimators':[50,100,200],
             #   'learning_rate':[0.01,0.1,0.2],
              #  'max_depth':[3,5,7]
            #}
        }

        for i in range(len(models)):
            model=list(models.values())[i]
            param=params[list(models.keys())[i]]

            if param:
                rs_model=RandomizedSearchCV(model,param,cv=5,n_iter=10,scoring='accuracy',n_jobs=-1,random_state=42)
                rs_model.fit(x_train,y_train)
                best_model=rs_model.best_estimator_
            else:
                model.fit(x_train,y_train)
                best_model=model

            y_pred=best_model.predict(x_test)
            acc=accuracy_score(y_test,y_pred)
            print(f"Model: {list(models.keys())[i]}")
            print(f"Accuracy: {acc}")
            print("Classification Report:")
            print(classification_report(y_test,y_pred))
            print("Confusion Matrix:")
            print(confusion_matrix(y_test,y_pred))
            print("-"*50)

            if acc>best_score:
                best_score=acc
                best_model_name=list(models.keys())[i]
                best_model_obj=best_model

        

        print("\nBest Model Summary")
        print("-" * 50)
        print(f"Best Model: {best_model_name}")
        print(f"Best Accuracy: {best_score}")

        
        with open("artifacts/best_model.pkl", "wb") as f:
                pickle.dump(best_model_obj, f)

        return best_model_obj, best_model_name, best_score


if __name__ == "__main__":
        vectorizer = pickle.load(open("artifacts/vectorizer.pkl", "rb"))
        train_df = pd.read_csv("artifacts/train.csv")
        test_df = pd.read_csv("artifacts/test.csv")

        x_train = vectorizer.transform(train_df["text"])
        x_test = vectorizer.transform(test_df["text"])
        y_train = train_df["output"]
        y_test = test_df["output"]

        obj = ModelTraining()
        obj.train_model(x_train, y_train, x_test, y_test)      
            