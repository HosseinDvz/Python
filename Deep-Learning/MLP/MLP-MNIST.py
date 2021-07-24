# -*- coding: utf-8 -*-
"""
@author: Hossein
"""

import keras
from keras.datasets import mnist
from keras.layers import Dense,Flatten 
from keras.models import Sequential
#from keras import initializers
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = mnist.load_data()


# Convert to "one-hot" vectors using the to_categorical function
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_train[0]

image_size = 784 # 28*28
num_classes = 10 # ten unique digits

def create_dense(layer_sizes):
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(units=layer_sizes[0],activation='relu', input_shape=(image_size,)))

    for s in layer_sizes[1:]:
        model.add(Dense(units = s, activation = 'relu'))

    model.add(Dense(units=num_classes, activation='softmax'))
    return model

def evaluate(model, batch_size=128, epochs=5):
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=.1, verbose=False)
    model.summary()
    loss, accuracy  = model.evaluate(x_test, y_test, verbose=False)
    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='best')
    plt.show()

    print()
    print(f'Test loss: {loss:.3}')
    print(f'Test accuracy: {accuracy:.3}')


exp = input("Choose an experiment:\n 1: changing depth (layers), fixed width(nodes per layer)\n 2:fixed depth changing width\n 3:change width and depth(needs more time)\n Enter 1-2-3:")

#experiments:
    
def experiment_1():
    for layers in range(1, 5):
        model = create_dense([256] * layers)
        evaluate(model)
        
def experiment_2():
    for nodes in [32, 64, 128, 256, 512, 1024, 2048]:
        model = create_dense([nodes])
        evaluate(model)


def experiment_3():
    for nodes_per_layer in [32, 128, 512]:
        for layers in [1, 3, 4, 5]:
            model = create_dense([nodes_per_layer] * layers)
            evaluate(model, epochs=10*layers)


if exp == "1":
    experiment_1()
elif exp=="2":
    experiment_2()
else:
    experiment_3()





