import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

import os
import json
import datetime
import pickle
import numpy
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
        train_data, train_labels, test_data, test_labels = self.loadData()
        print('Preprocessing Data - ' + self.timestamp())
        clean_train, clean_test = self.preprocessData(train_data, test_data)
        print('Extracting Features - ' + self.timestamp())
        train_vectors, test_vectors = self.extractFeatures(clean_train, clean_test)
        print('Training Model - ' + self.timestamp())
        model = self.fitModel(train_vectors, train_labels)
        print('Evaluating Model - ' + self.timestamp())
        self.evaluate(model, test_vectors, test_labels)

    def loadData(self):
        # Load the data as specified by the config file
        config_data = self.config  # Assuming config is a list with a single dictionary
        dataLoader = self.resolve('dataLoader', config_data['dataLoader'])()
        return dataLoader.load(config_data['dataPath'])

    def preprocessData(self, train_data, test_data):
        # Preprocess the data as specified in the config file
        config_data = self.config  # Assuming config is a list with a single dictionary
        preprocessor = Preprocessor()

        if path.exists(config_data['preprocessingPath'] + "Wili1.pickle"):
            with open(config_data['preprocessingPath'] + "Wili1.pickle", "rb") as file:
                train_data, test_data = pickle.load(file)
        else:
            for step in config_data['preprocessing']:
                train_data = preprocessor.process(step, train_data)
                test_data = preprocessor.process(step, test_data)
            with open(config_data['preprocessingPath'] + "Wili1.pickle", "wb+") as file:
                pickle.dump((train_data, test_data), file)

        return train_data, test_data

    def extractFeatures(self, train_data, test_data):
        # Extract features and pass them as concatenated arrays
        config_data = self.config # Assuming config is a list with a single dictionary
        fe = FeatureExtractor(config_data['features'], config_data['featurePath'], config_data['featureKwargs'])
        fe.buildVectorizer(train_data)

        train_vectors = fe.process(train_data)
        test_vectors = fe.process(test_data)

        if len(train_vectors) > 1:
            train_vectors = numpy.concatenate(train_vectors, axis=1)
        else:
            train_vectors = train_vectors[0]
        if len(test_vectors) > 1:
            test_vectors = numpy.concatenate(test_vectors, axis=1)
        else:
            test_vectors = test_vectors[0]

        return train_vectors, test_vectors

    def fitModel(self, train_vectors, train_labels):
        # Convert train_vectors to dense numpy array if it's a sparse matrix
        if isinstance(train_vectors, scipy.sparse.csr.csr_matrix):
            train_vectors = train_vectors.toarray()

        # Ensure train_labels are numpy array of appropriate type (e.g., int for classification)
        train_labels = self.encode_labels(train_labels)

        # Assuming model creation and loading logic is handled directly here
        model_path = os.path.join(self.config['modelPath'], "model.h5")

        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
        else:
            model = self.create_tf_model(train_vectors)

            # Debugging prints to check data types
            print("train_vectors shape:", train_vectors.shape, "dtype:", train_vectors.dtype)
            print("train_labels shape:", train_labels.shape, "dtype:", train_labels.dtype)

            model.fit(train_vectors, train_labels, epochs=10, batch_size=32, validation_split=0.2)
            model.save(model_path)

        return model

    def encode_labels(self, labels):
        # Example function to encode categorical labels to numeric format
        label_map = {'negative': 0, 'positive': 1}  # Example mapping for binary classification
        return numpy.array([label_map[label] for label in labels])

    def create_tf_model(self, train_vectors):
        model = Sequential([
            Dense(128, activation='relu', input_shape=(train_vectors.shape[1],)),  # Adjust input shape here
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        return model


    def evaluate(self, model, test_data, test_labels):
        config_data = self.config  # Assuming config is a list with a single dictionary

        if path.exists(config_data['metricPath'] + "metrics.pickle"):
            with open(config_data['metricPath'] + "metrics.pickle", "rb") as file:
                results = pickle.load(file)
        else:
            # Ensure test_data is converted to a dense format if it's sparse
            if isinstance(test_data, scipy.sparse.csr.csr_matrix):
                test_data = test_data.toarray()

            # Assuming test_labels are string labels like "positive", "negative", etc.
            label_mapping = {"positive": 1, "negative": 0}

            # Convert test_labels to numeric labels
            numeric_test_labels = [label_mapping[label] for label in test_labels]

            predictions = model.predict(test_data)
            predictions = (predictions > 0.5).astype(int)
            
            results = {}
            for metric in config_data['metrics']:
                results[metric] = self.resolve('metrics', metric)(numeric_test_labels, predictions, **config_data['metricsKwargs'][config_data['metrics'].index(metric)])
            
            with open(config_data['metricPath'] + "metrics.pickle", "wb+") as file:
                pickle.dump(results, file)

        self.output(results)
        print(results)



    def output(self, results):
        config_data = self.config  # Assuming config is a list with a single dictionary
        output_file = os.path.join(config_data['outputPath'], config_data['experimentName'])
        with open(output_file, 'w') as F:
            F.write(json.dumps(config_data) + '\n')
            for metric in results:
                F.write(metric + ',%f\n' % results[metric])

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

    def timestamp(self):
        return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    def predict(self, data):
        # Ensure data is preprocessed and feature-extracted like training data
        clean_data = self.preprocessData(data, data)[0]
        vectors = self.extractFeatures(clean_data, clean_data)[0]

        # Convert vectors to dense format if necessary
        if isinstance(vectors, scipy.sparse.csr.csr_matrix):
            vectors = vectors.toarray()

        # Load the model
        model_path = os.path.join(self.config['modelPath'], "model.h5")
        model = tf.keras.models.load_model(model_path)

        # Make predictions
        predictions = model.predict(vectors)
        predictions = (predictions > 0.5).astype(int)

        # Decode numeric predictions to labels
        label_map = {0: 'negative', 1: 'positive'}
        decoded_predictions = [label_map[pred[0]] for pred in predictions]
        return decoded_predictions
