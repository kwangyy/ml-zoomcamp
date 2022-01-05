#!/usr/bin/env python
# coding: utf-8

# ## Steps for our model:
# 1. Import keras from tensorflow (as tf)
# 2. Load our model
# 3. Save our model as a folder using `tf.saved_model.save()`
# - e.g. `tf.saved_model.save(model, 'model-name')` where model-name can be changed (i.e. 'clothing-model')
# 4. In our command line, we can use `ls -lhR model-name` to check the contents of our folder
# - Make sure that there is a saved_model.pb file in there, alongside an 'assets' folder and a 'variables' folder
# - The 'variables' folder should include variables.data-00000-of-00001 and variables.index
# 5. Type in command line `saved_model_cli show --dir model-name --all` and get the inputs and outputs from the saved_model
# - i.e. inputs and outputs under `signature_def['serving_default']`


import os
import grpc 

import tensorflow as tf

from flask import Flask
from flask import request 
from flask import jsonify 

from proto import np_to_protobuf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

 
# From the initialization of Docker 
host = os.getenv('TF_SERVING_HOST', 'localhost:8500')

# We use an insecure channel because we are running this locally 
channel = grpc.insecure_channel(host)

# Stub used to make predictions 
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

from keras_image_helper import create_preprocessor 

preprocessor = create_preprocessor('xception', target_size = (299,299))

classes = [
    'dress',
    'hat',
    'longsleeve',
    'outwear',
    'pants',
    'shirt',
    'shoes',
    'shorts',
    'skirt',
    't-shirt'
]

def prepare_request(X):
	pb_request = predict_pb2.PredictRequest()

	pb_request.model_spec.name = 'clothing-model'
	pb_request.model_spec.signature_name = 'serving_default'

	pb_request.inputs['input_8'].CopyFrom(np_to_protobuf(X))
	return pb_request


def prepare_response(pb_response):
	preds = pb_response.outputs['dense_7'].float_val
	return dict(zip(classes, preds))


def predict(url):
	X = preprocessor.from_url(url)
	pb_request = prepare_request(X)
	pb_response = stub.Predict(pb_request, timeout=20)
	response = prepare_response(pb_response)
	return response

app = Flask('gateway')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
	data = request.get_json()
	url = data['url']
	result = predict(url)
	return jsonify(result)

if __name__ == '__main__':
	url = 'http://bit.ly/mlbookcamp-pants'
	response = predict(url)
	print(response)
	# app.run(debug = True, host = '0.0.0.0', port=9696)
