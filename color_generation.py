# -*- coding: utf-8 -*-
# Color_generation

# Import modules
from flask import Flask, jsonify, request
import pandas as pd
from absl import logging
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns

app = Flask(__name__)

# Change path of excel file if needed
keywords_list = pd.read_excel('Keywords_List.xlsx')
key_words = keywords_list['Keywords_list']

model = hub.load('Model/')

def embed(input):
  return model(input)

# Change the path for API call in case needed
@app.route('/', methods=['POST'])
def similarity():
  app_name = request.json.get('app_name',None)
  app_desc = request.json.get('app_desc',None)

  # Dictionary to return
  response = {}

  # Finding similarity here and return
  desc_embedding = embed(app_desc)
  keyword_embedding = embed(key_words)
  corr = np.inner(desc_embedding,keyword_embedding)

  keywords_dict = dict(zip(list(key_words), list(corr)))

  if corr.shape[0] > 0:
    success = 0
  else:
    success = -1

  # Create reponse dict here
  response['success'] = success
  response['app_name'] = app_name
  response['key_words'] = keywords_dict

  return jsonify(response)

if __name__ == '__main__':
  
    app.run(host="139.59.21.103", port=8000, debug=True)
