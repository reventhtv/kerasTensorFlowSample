# This is a simple keras with TensorFlow Python script.

#Data Preparation and Processing
import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

train_labels = []
train_samples = []

#Example data:
#An experimental drug was tested on individuals from ages 13 to 100 in a clinical trial
#The trial had 2100 participants. Half were under 65 years old, half were 6 years or older
#Around 95% of patients 65 or older experienced side effects
#Around 95% of patients under 65 experienced no side effects

for i in range(50):
    #The ~5% of younger individuals who experienced side effects
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(1)

    #The ~5% of older individuals who didn't experience side effects
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(0)


for i in range(1000):
    #The ~95% of younger individuals who did not experience side effects
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(0)

    #The ~95% of older individuals who experienced side effects
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(1)

#for i in train_samples:
#    print(i)

#for i in train_labels:
#    print(i)

train_labels = np.array(train_labels)
train_samples = np.array(train_samples)
#We shuffle function to get rid of any imposed order from the data generation process
train_labels, train_samples = shuffle(train_labels, train_samples)
#We are scaling our dataset to normalize/standardize for a quicker training of our NN and more efficient
scaler = MinMaxScaler(feature_range=(0,1))
#Since fit_transform function wont accept 1D data we are using reshape function here. Just a formality
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1,1))

#for i in scaled_train_samples:
#    print(i)

#simple ANN sequential model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
#Below two used while training the model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy

# Below snippet will try to identify tensorflow GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices)) #will return 0 because i am not running GPU at this moment
#tf.config.experimental.set_memory_growth(physical_devices[0], True)


#Building sequential model. Sequential model is Linear stack of layers
#units in Dense function also known as nodes/neurons
#last layer(Dense) is our output layer. Softmax gives probabilities of each output class
model = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax')
])

model.summary() #Prints out the visual summary of the architecture of the model we just created


#Now we will  train our ANN model

#Preparing the model for training. Gets everything in order before we train the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#Training occurs whenever we call this fit function
model.fit(x=scaled_train_samples, y=train_labels, batch_size= 10, epochs=30, shuffle=True, verbose=2)
#epochs = 30 means model is going to process/train 30 times in all of the dataset completing before training process


#Validation set: Here loss and accuracy is predicted on validated set. How well our model is accurate.
#There is another way to use validation by using argument "validation_data=valid_set" where valid_set = (x_val, y_val) of numpy arrays or tensors
#where x_val is a numpy array or tensor containing validation samples and y_val is a numpy array or tensor containing validation labels
#whenever we use below method 10%(valid_set = 0.1) of validation set is held out of training set.
# So moved data will no longer be processed for training. Eventhough we call shuffle=True it is set to True by default.
#Whenever we call validation_split in this way the split occurs before the training set is shuffled.
#It is important to know that valdiation_split is last x% of the training set. Therefore may not be shuffled.
model.fit(x=scaled_train_samples, y=train_labels, validation_split=0.1, batch_size=10, epochs=30, shuffle=True, verbose=2)

#In this example our model is not overfitting. It is performing well