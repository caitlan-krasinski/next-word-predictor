import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import tensorflow as tf
import keras
from keras.models import load_model
import tensorflow_hub as hub

model = None
app = Flask(__name__)

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

words = pd.read_csv("words.csv")
words = list(words.words)


def load_model():
    global model
    model = keras.models.load_model('next_word_model.h5')


@app.route('/')
def my_form():
    return render_template('user-form.html', original='', predict='')


@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    prediction = model.predict(x=embed([text]).numpy())
    index = np.argmax(prediction[-1])
    pred = words[index]
    return render_template('user-form.html', original=text, predict=pred)

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=80)
