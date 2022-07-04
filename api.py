from flask import Flask, request, jsonify
from flask_restful import Api, Resource, reqparse
import numpy as np

from transformers import AutoTokenizer
import numpy as np
from transformers import TFAutoModelForSequenceClassification
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
import pandas as pd
from chartbot_config import *


app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('data')

df = pd.read_csv(raw_data_path, encoding = 'unicode_escape')
        
num_labels = df.Intent.nunique()
df = df.melt(id_vars=["Domain", "Sub domain", "Intent", "Answer Format"]).drop("variable", axis = 1)
Classes_dict_1 = dict(zip(list(df["Answer Format"].unique()), [i for i in range(df["Intent"].nunique())]))
Classes_dict = {value:key for key, value in Classes_dict_1.items()}
        
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels, problem_type="multi_label_classification")
model.load_weights(model_path)
    
opt = Adam()
loss = BinaryCrossentropy(from_logits=True)
    
model.compile(
    optimizer = opt,
    loss = loss,
    metrics=["accuracy"],
)

# Define how the api will respond to the post requests
class QuestionClassifier(Resource):
    @app.route('/foo', methods=['POST']) 
    def post():
    
        data = request.json
        sentence = data['data']
        tokenized_dataset = tokenizer(sentence,  padding=True, truncation=True, return_tensors = "tf")
        output = model(**tokenized_dataset)["logits"]
        class_preds = [np.argmax(i) if np.sum(i) == 1 else len(i) for i in output > 0][0]
        
        return jsonify(data=Classes_dict.get(class_preds, "Sorry I cannot understand."))

api.add_resource(QuestionClassifier, '/chartbot')
