import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

import os
import json
import datetime
import pickle
import numpy as np
import scipy  
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from dataLoader import DataLoader
from csvLoader import CSVLoader
from jsonLoader import JSONLoader
from preprocessor import Preprocessor
from featureExtractor import FeatureExtractor
from os import path

class Pipeline(object):

    def __init__(self, configFile):
        # Load the specific configuration file
        self.config = json.load(open(configFile, 'r'))

    def execute(self):
        # Execute the pipeline
        print('Loading Data - ' + self.timestamp())
        data_to_predict = self.loadRealTimeData()
        print('Preprocessing Data - ' + self.timestamp())
        clean_data = self.preprocessData(data_to_predict)
        print('Extracting Features - ' + self.timestamp())
        vectors = self.extractFeatures(clean_data)
        print('Predicting with Model - ' + self.timestamp())
        predictions = self.predict(vectors)
        print('Outputting Results - ' + self.timestamp())
        self.output(predictions)

    def loadRealTimeData(self):
        # Load real-time data as specified by the config file
        # Example implementation, modify as per your data loading mechanism
        # Here we assume it's a single piece of data or a batch that needs processing
        return ["Your real-time data here"]

    def preprocessData(self, data_to_predict):
        # Preprocess the real-time data as specified in the config file
        preprocessor = Preprocessor()

        processed_data = data_to_predict  # Modify this based on your preprocessing logic

        return processed_data

    def extractFeatures(self, clean_data):
        # Extract features for the real-time data
        fe = FeatureExtractor(self.config['features'], self.config['featurePath'], self.config['featureKwargs'])
        fe.buildVectorizer(clean_data)
        vectors = fe.process(clean_data)

        if len(vectors) > 1:
            vectors = np.concatenate(vectors, axis=1)
        else:
            vectors = vectors[0]

        return vectors

    def predict(self, vectors):
        # Load the pre-trained model
        model_path = os.path.join(self.config['modelPath'], "model.h5")
        model = load_model(model_path)

        # Make predictions
        predictions = model.predict(vectors)
        predictions = (predictions > 0.5).astype(int)

        # Example: Decoding predictions to labels
        label_map = {0: 'negative', 1: 'positive'}
        decoded_predictions = [label_map[pred[0]] for pred in predictions]

        return decoded_predictions

    def output(self, predictions):
        # Output predictions to a file or any other desired format
        output_file = os.path.join(self.config['outputPath'], self.config['experimentName'] + "_predictions.txt")
        with open(output_file, 'w') as f:
            for prediction in predictions:
                f.write(prediction + '\n')

    def timestamp(self):
        return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def resolve(self, category, setting):
        # Resolve a specific config string to function pointers or list thereof
        configurations = {'dataLoader': {'baseLoader': DataLoader,
                                         'CSVLoader': CSVLoader,
                                         'JSONLoader': JSONLoader},
                          'metrics': {'accuracy': accuracy_score,
                                      'f1': f1_score,
                                      'recall': recall_score,
                                      'precision': precision_score}
                          }
        assert setting in configurations[category]
        return configurations[category][setting]

if __name__ == '__main__':
    p = Pipeline('config.json')
    p.execute()
