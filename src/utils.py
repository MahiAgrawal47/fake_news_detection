import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer



from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def text_cleaning(text):
     lemmatizer=WordNetLemmatizer()

     # Remove source/dateline artifacts
     text = re.sub(r'^[A-Za-z\s,]+\([A-Za-z\s]+\)\s*-\s*', '', text)
     text = re.sub(r'\(Reuters\)|\(AP\)|\(Bloomberg\)', '', text, flags=re.IGNORECASE)
     text = re.sub(r'\bReuters\b', '', text, flags=re.IGNORECASE)

     # Remove image credit boilerplate
     text = re.sub(r'featured image.*?(via)?\s*(getty images?)?', '', text, flags=re.IGNORECASE)
     text = re.sub(r'\bgetty images?\b', '', text, flags=re.IGNORECASE)

     # Remove site/source names
     text = re.sub(r'\b(breitbart|wfb|nyp|cnn|fox news|msnbc)\b', '', text, flags=re.IGNORECASE)

     # Remove blog-style CTAs
     text = re.sub(r'\b(read more|watch|click here|via)\b', '', text, flags=re.IGNORECASE)

     text=text.lower()
     text=re.sub('[^a-zA-Z0-9]',' ',text)
     text=text.split()
     stop_words = set(stopwords.words('english'))
     text=[lemmatizer.lemmatize(word) for word in text if word not in stop_words]
     text=' '.join(text)
     return text


def get_vectorizer():
    return TfidfVectorizer(max_features=50000, ngram_range=(1,2),stop_words='english',
    lowercase=True)

def fit_vectorizer(vectorizer, corpus):
    return vectorizer.fit_transform(corpus)

def transform_vectorizer(vectorizer, corpus):
    return vectorizer.transform(corpus)



    
    

