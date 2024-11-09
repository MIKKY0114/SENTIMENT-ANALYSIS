import pandas as pd
import numpy as np
import re
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from transformers import BertTokenizer, BertModel
import torch
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from sklearn.preprocessing import StandardScaler
import joblib
import cloudpickle

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, stopwords):
        self.stopwords = stopwords
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")

    def clean_text(self, text):
        text = re.sub(r"\d+", "", text)
        text = re.sub(r"[^\w\s]", "", text)
        text = text.lower()
        return text
    
    def remove_stopwords(self, text):
        tokens = text.split()
        filtered_tokens = [word for word in tokens if word not in self.stopwords]
        return " ".join(filtered_tokens)
    
    def bert_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            output= self.bert_model(**inputs)
        embedding = output.last_hidden_state.mean(dim=1).numpy()
        return embedding.flatten()
    
    def transform(self, X, y=None):
        processed_text = []
        for text in X:
            cleaned = self.clean_text(text)
            no_stopword = self.remove_stopwords(cleaned)
            bert_embedded = self.bert_embedding(no_stopword)
            processed_text.append(bert_embedded)
        return np.array(processed_text)
    
    def fit(self, X, y=None):
        return self
    
stopwords = set(stopwords.words("english")) - {"not","no","never"}

pipeline = Pipeline([
    ("preprocessor", TextPreprocessor(stopwords=stopwords)),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression())
])

df = pd.read_excel(r"C:\Users\701540\VS_PY\SENTI\sentiment.xlsx")

df.sample(2)

X_train = df["REVIEW"]
y_train = df["sentiment_label"]

pipeline.fit(X_train, y_train)

with open("pipeline.pkl", "wb") as f:
    cloudpickle.dump(pipeline, f)

with open("pipeline.pkl", "rb") as f:
    pipeline = cloudpickle.load(f)