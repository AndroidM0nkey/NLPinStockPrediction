from fastapi import FastAPI
import pickle
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.corpus import stopwords

app = FastAPI()

nltk.download('stopwords')

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

class Item(BaseModel):
    title: str

def clean(headline):
    headline = headline.lower()
    headline = re.sub('[^a-z A-Z 0-9-]+', '', headline)
    headline = " ".join([word for word in headline.split() if word not in stopwords.words('english')])
    return headline

@app.post('/predict')
def predict(item: Item):
    new_title = clean(item.title)
    vectorized_title = vectorizer.transform([new_title])
    prediction = model.predict(vectorized_title)
    return {'prediction': prediction[0]}
