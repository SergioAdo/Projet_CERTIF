
from __future__ import division, print_function
# coding=utf-8
import sys
import os
# os.environ['KERAS_BACKEND'] = 'theano'
import keras as ks
import glob
import re
import numpy as np
import tensorflow as tf
from keras.applications import VGG16

# Keras
from keras.models import load_model
from keras.preprocessing import image


from keras.backend import set_session

# Pytesseract
from PIL import Image
import pytesseract
#import cv2
#import ftfy


# Flask utils
from flask import Flask, redirect, url_for, request, render_template, jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import json

app = Flask(__name__)

sess = tf.Session()
graph = tf.get_default_graph()

set_session(sess)
model = tf.keras.models.load_model('C:/Users/tripl/PROJETCERTIF_IA/IHM/model/20200330_vgg16_94.h5', compile=False)
model.summary()

# conv_base = VGG16(weights='imagenet',
#                  include_top=False,
#                  input_shape=(224, 224,3))


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global sess
    global graph
    with graph.as_default():
        set_session(sess)
        f = request.files['file']
        img = tf.keras.preprocessing.image.load_img(f, target_size=(224, 224))
        img = np.expand_dims(img, axis=0)
        #img /= 255.  # On normalize
        # Extraction des  features
        #features = conv_base.predict(x.reshape(1,224, 224, 3))
        # On predit
        result = (model.predict(img)*100).tolist()

        print("Malaria:", result[0][0], " Sain:", result[0][1])

        return json.dumps(result)
        #Creation dict
        # dic = {}
        # dic['Indice:'] = classes.tolist()

        # print (dic)
        # print(pytesseract.image_to_string(Image.open(f)))

        # indice= json.dumps(dic)
        # if classes > 0.5 :
        #     print('Inconnu ou CNI de mauvaise qualité')

        # else:
        #     print('CNI')

        #     # infos= json.dumps(pytesseract.image_to_string(Image.open(f)))
        #     # text= ftfy.fix_text(infos)
        #     # text= ftfy.fix_encoding(text)

        #     return '{}'.format(classes)


if __name__ == '__main__':
    app.run(debug=True)
