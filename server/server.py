from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

import pickle
import re
import nltk
from nltk.corpus import stopwords

app = FastAPI()

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import keras

from tensorflow.keras.models import load_model

import numpy as np


with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)


custom_objects = {'KerasLayer': hub.KerasLayer} 

with keras.utils.custom_object_scope(custom_objects):
    dl_model = load_model('dl_model_2.h5')


def clean(headline):
    headline = headline.lower()
    headline = re.sub('[^a-z A-Z]+', '', headline)
    headline = " ".join([word for word in headline.split() if word not in stopwords.words('english')])
    return headline


def predict_ml_model(texts):

    processed_texts = []
    for text in texts:
        processed_texts.append(clean(text))

    vectorized_titles = vectorizer.transform(processed_texts)
    prediction = model.predict(vectorized_titles)
    return prediction


classes = ['Negative', 'Neulral', 'Positive']

def predict_dl_model(texts):
    prediction = dl_model.predict(texts)
    return prediction 


class Text(BaseModel):
    text: str

class Texts(BaseModel):
    texts: List[str]

ml_stats = {}
dl_stats = {}

@app.get("/predict_ml/")
def predict_ml(text: str):
    result = predict_ml_model([text])
    ml_stats[result[0]] = ml_stats.get(result[0], 0) + 1
    return {"class": result[0]}

@app.post('/predict_batch_ml/')
async def predict_batch_ml(data: Texts):
    results = predict_ml_model(data.texts)
    ans = []
    for result in results:
         ml_stats[result] = ml_stats.get(result, 0) + 1
         ans.append(result)

    return {"labels": ans}

@app.get("/stats_ml/")
def stats_ml():
    return ml_stats

@app.get("/predict_dl/")
def predict_dl(text: str):
    result = predict_dl_model([text])
    dl_stats[result[0]] = dl_stats.get(result[0], 0) + 1
    return {"class": result[0]}

@app.post('/predict_batch_dl/')
async def predict_batch_dl(data: Texts):
    results = predict_dl_model(data.texts)
    ans = []

    for result in results:
        res = classes[np.argmax(result)]
        dl_stats[res] = dl_stats.get(res, 0) + 1
        ans.append(res)

    return {"labels": ans}

@app.get("/stats_dl/")
def stats_dl():
    return dl_stats
