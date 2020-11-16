#preprocess test data
#creating a testset and we will use our model to inference on it

import numpy as np
from random import randint
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
#import tensorflow as tf
#from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
#Below two used while training the model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy

test_labels = []
test_samples = []

for i in range(10):
    #The ~5% of younger individuals who experienced side effects
    random_younger = randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(1)

    #The ~5% of older individuals who didn't experience side effects
    random_older = randint(65,100)
    test_samples.append(random_older)
    test_labels.append(0)


for i in range(200):
    #The ~95% of younger individuals who did not experience side effects
    random_younger = randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(0)

    #The ~95% of older individuals who experienced side effects
    random_older = randint(65,100)
    test_samples.append(random_older)
    test_labels.append(1)


test_labels = np.array(test_labels)
test_samples = np.array(test_samples)
test_labels, test_samples = shuffle(test_labels, test_samples)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_test_samples = scaler.fit_transform(test_samples.reshape(-1, 1))


model = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax')
])

#Predicting model
predictions = model.predict(x=scaled_test_samples, batch_size=10, verbose=0)

for i in predictions:
    print(i)
#Interpreting the return i values. For example [0.43515587 0.56484413] represents 43% probability to 'x' patient not experiencing side effects
#Around 56% probability of the patient experiencing a side effect

rounded_predictions = np.argmax(predictions, axis=-1)

for i in rounded_predictions:
    print(i)