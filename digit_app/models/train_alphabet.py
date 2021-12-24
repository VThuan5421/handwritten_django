"""
path = "c:/vs_data/handwritten/A_Z Handwritten Data.csv" # Absolute path
# download: https://www.kaggle.com/sachinpatel21/az-handwritten-alphabets-in-csv-format

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import cv2

data = pd.read_csv(path).astype('float32')
# The dataset has 785 columns with the first column being the label,
# The remaining 784 columns is a 784 length vector.
# We have to reshape a vector into a 28x28 numpy array. (784 = 28 x 28)
X = data.drop('0', axis = 1)
y = data['0']

word_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',
    10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',
    20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'}
# split data to training and test data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
# Reshaping the training and test data so that can be displayed as an image
x_train = np.reshape(x_train.values, (x_train.shape[0], 28, 28))
x_test = np.reshape(x_test.values, (x_test.shape[0], 28, 28))

# Reshape the training and test data so that it can be put into the model
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

# because expect output is a number in range(26) and value in each row
# of label datasets (y_train, y_test) is a number in range(26). We should
# make each row of labels to an one-host array.
# Example. The output is 3 => one-host array is [0. 0. 0. 1. 0. ... 0. 0.].
# Length array = 26. The output is 5 => onehost array is [0. 0. 0. 0. 0. 1. 0. ... 0. 0.]
# Use to_categorical
y_train_one = to_categorical(y_train, num_classes = 26, dtype = 'int')
y_test_one = to_categorical(y_test, num_classes = 26, dtype = 'int')

def save_some_handwritting():
    """ Function to save some image to predict."""
    for i in range(100):
        num = np.random.randint(0, len(x_test))
        filepath = "_" + str(i) + ".jpg"
        cv2.imwrite(str(word_dict[np.argmax(y_test_one[num])]) + filepath, x_test[num])
    print("Done!")
#save_some_handwritting()

# Define the CNN model
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', input_shape = (28, 28, 1)))
model.add(MaxPooling2D(pool_size = (2, 2), strides = 2))

model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
model.add(MaxPooling2D(pool_size = (2, 2), strides = 2))

model.add(Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu', padding = 'valid'))
model.add(MaxPooling2D(pool_size = (2, 2), strides = 2))

model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))

model.add(Dense(26, activation = 'softmax'))

# compiling and fitting the model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
hist = model.fit(x_train, y_train_one, epochs = 3, batch_size = 128, verbose = 1)
model.save('alphabetmodel.h5')
score = model.evaluate(x_test, y_test_one)
print("Loss: ", score[0])
print("Accuracy: ", score[1])
"""



