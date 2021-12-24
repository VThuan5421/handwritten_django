from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Dropout, Flatten, Conv2D
from keras.utils.np_utils import to_categorical


# Load data from keras datasets

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("All shape: ", x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# fit data
# normalizing data from 0-255 to 0-1 by divide each to 255
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_train = x_train / 255.0
x_train = x_train.astype('float32')

x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_test = x_test / 255.0
x_test = x_test.astype('float32')

# because output value belong to ten class [0, 9], we group the result to ten classes using to_categorical
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Create CNN model (Convolutional neural network)
model = Sequential()
# input layer
model.add(Conv2D(32, kernel_size = (5, 5,), activation = 'relu', input_shape = (28, 28, 1)))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())

model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.3))

model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(10, activation = 'softmax'))
model.summary()

# Compile and train the model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(x_train, y_train, batch_size = 128, epochs = 10, verbose = 1)
predicted = model.predict(x_test)

score = model.evaluate(x_test, y_test, verbose = 0)
print("Test loss: ", score[0])
print("Test accuracy: ", score[1])
model.save('digitmodel.h5')

print("Model saved succesfully.")

# Please cd to ./path/to/models and run file train_model.py: python train_model.py

