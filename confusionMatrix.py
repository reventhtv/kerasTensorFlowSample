#matplotlib inline
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from testModel import *

#Visualizing NN during inference

cm = confusion_matrix(y_true=test_labels, y_pred=rounded_predictions)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting 'normalize=True'
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix without normalization")

    print(cm)

    thresh = cm.max() / 2.
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
cm_plot_labels = ['no_side_effects', 'had_side_effects']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')


##Save and load a model
#model.save()
#Checks first to see if file exists already. If not, the model is saved to disk
import os.path

if not os.path.exists('models'):
    os.makedirs('models')

if os.path.isfile('models/medical_trial_model.h5') is False:
    model.save('models/medical_trial_model.h5')

#The save function saves following:
#The architecture of the model, allowing to re create the model
#weights of the model
#training configuration (loss, optimizer)
#state of the optimizer, allowing to resume training excatly where you left off

from tensorflow.keras.models import load_model
new_model = load_model('models/medical_trial_model.h5', compile=True)

#new_model.summary()
#new_model.get_weights()
#new_model.optimizer


#model.to_json()
#If we wants to save only model architecture by saving it to .json string
json_string = model.to_json()

#print(json_string)
# To create a new model with older version architectures we can import .json string

from tensorflow.keras.models import model_from_json
model_architecture = model_from_json(json_string)

#model_architecture.summary()

#We can only save weights of a model by following function
if os.path.isfile('models/my_model_weights.h5') is False:
    model.save_weights('models/my_model_weights.h5')

model2 = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax')
])

model2.load_weights('models/my_model_weights.h5')

model2.get_weights()