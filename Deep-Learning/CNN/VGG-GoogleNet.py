# -*- coding: utf-8 -*-
"""
@author: Hossein
"""


from keras.models import Model
from keras.optimizers import SGD#, Adam
from keras.datasets import cifar10,mnist
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D,AveragePooling2D, Flatten, Dense, Dropout
#from keras.utils import plot_model
from keras.regularizers import l2
import matplotlib.pyplot as plt

dataset=input("Choose the data set for training, 1: mnist    2:cifar10\n Enter 1 or 2:")
model_type=input("Choose the model type for trainin:\n 1: VGG\n 2: GoogelNet \n 3: ResNet\n Enter 1 or 2 or 3:")
epo=int(input("Enter the number of Epochs for training the model: "))


def load_data(x):
    if x=='1':
        (x_train,y_train) , (x_test,y_test) = mnist.load_data()
        #we need 4 dims for mnist data set so I reshape it and one dimension and normalized
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)/255.0
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)/255.0

    else:
        (x_train,y_train) , (x_test,y_test) = cifar10.load_data()
        x_train = x_train/255.0
        x_test = x_test/255.0
    return x_train,y_train,x_test,y_test

def get_input_shape():
    if dataset == '1':
        inp = Input(shape=(28, 28, 1))
    else:
        inp = Input(shape=(32, 32, 3))
    return inp
    

# function for creating a vgg block
def vgg_block(layer_in, n_filters, n_conv):
	# add convolutional layers
	for _ in range(n_conv):
		layer_in = Conv2D(n_filters, (3,3), padding='same', activation='relu')(layer_in)
	# add max pooling layer
	layer_in = MaxPooling2D((2,2))(layer_in)
	return layer_in


def VGG():
    # define model input
    inp = get_input_shape()
    # add  three vgg modules
    layer = vgg_block(inp, 32, 2)
    layer = vgg_block(layer, 64, 2)
    layer = vgg_block(layer, 128, 2)
    
    # Flattening
    flat = Flatten()(layer)
    dense = Dense(128, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(0.001))(flat)
    #drop = Dropout(0.1)(dense)
    output = Dense(10, activation = 'softmax')(dense)
    model = Model(inputs=inp, outputs=output)
    # summarize model
    model.summary()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model



from keras.layers.merge import concatenate

# function for creating a projected inception module
def inception_module(layer_in, f1, f2_in, f2_out, f3_in, f3_out, f4_out):
	# 1x1 conv
	conv1 = Conv2D(f1, (1,1), padding='same', activation='relu')(layer_in)
	# 3x3 conv
	conv3 = Conv2D(f2_in, (1,1), padding='same', activation='relu')(layer_in)
	conv3 = Conv2D(f2_out, (3,3), padding='same', activation='relu')(conv3)
	# 5x5 conv
	conv5 = Conv2D(f3_in, (1,1), padding='same', activation='relu')(layer_in)
	conv5 = Conv2D(f3_out, (5,5), padding='same', activation='relu')(conv5)
	# 3x3 max pooling
	pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(layer_in)
	pool = Conv2D(f4_out, (1,1), padding='same', activation='relu')(pool)
	# concatenate filters, assumes filters/channels last
	layer_out = concatenate([conv1, conv3, conv5, pool], axis=-1)
	return layer_out


def GoogLeNet():
    # define model input
    inp = get_input_shape()
    # add inception block 1
    layer = inception_module(inp, 64, 96, 128, 16, 32, 32)
    # add inception block 1
    #layer = inception_module(layer, 128, 128, 192, 32, 96, 64)
    layer = AveragePooling2D((5,5))(layer)
    # create model
    
    flat = Flatten()(layer)
    dense = Dense(256, activation = 'relu')(flat)
    drop = Dropout(0.2)(dense)
    output = Dense(10, activation = 'softmax')(drop)
    model = Model(inputs=inp, outputs=output)
    # summarize model
    model.summary()
    #opt = SGD(learning_rate=0.01, momentum=0.15)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # plot model architecture
    return model


from keras.layers import Activation
from keras.layers import add


# function for creating an identity or projection residual module
def residual_module(layer_in, n_filters):
	merge_input = layer_in
	# check if the number of filters needs to be increase, assumes channels last format
	if layer_in.shape[-1] != n_filters:
		merge_input = Conv2D(n_filters, (1,1), padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
	# conv1
	conv1 = Conv2D(n_filters, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
	# conv2
	conv2 = Conv2D(n_filters, (3,3), padding='same', activation='linear', kernel_initializer='he_normal')(conv1)
	# add filters, assumes filters/channels last
	layer_out = add([conv2, merge_input])
	# activation function
	layer_out = Activation('relu')(layer_out)
	return layer_out

def ResNet():
    # define model input
    inp = get_input_shape()
    # add vgg module
    layer = residual_module(inp, 64)
    # create model
   
    flat = Flatten()(layer)
    dense = Dense(64, activation = 'relu')(flat)
    output = Dense(10, activation = 'softmax')(dense)
    model = Model(inputs=inp, outputs=output)
    # summarize model and compile
    model.summary()
    opt = SGD(learning_rate=0.01, momentum=0.5)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def run_experiment():
    x_train,y_train,x_test,y_test = load_data(dataset)
    if model_type=='1':
        model = VGG()
        #plot_model(model, show_shapes=True, to_file='multiple_vgg_blocks.png')
        history = model.fit(x_train,y_train, epochs=epo,validation_split=.1, batch_size=100)
    elif model_type=='2':
        model = GoogLeNet()
        #plot_model(model, show_shapes=True, to_file='GoogelNet.png')
        history = model.fit(x_train,y_train, epochs=epo,validation_split=.1, batch_size=100)
    else:
        model = ResNet()
        #plot_model(model, show_shapes=True, to_file='ResNet.png')
        history = model.fit(x_train,y_train, epochs=epo,validation_split=.1, batch_size=100)
    
    print("model accuracy= ",model.evaluate(x_test, y_test))
    return history


def result_plots(history):
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    x_axis=list(range(1,epo+1))
    plt.plot(x_axis,history.history['loss'], color='blue', label='train')
    plt.plot(x_axis,history.history['val_loss'], color='orange', label='validation')
    plt.xlim(1,epo)
    plt.xticks(range(1,epo+1))
    plt.legend()
    plt.savefig("mygraph.png")
    plt.show()
    
    plt.subplot(212)
    plt.title("Classification Accuracy")
    plt.plot(x_axis,history.history['accuracy'], color='blue', label='train')
    plt.plot(x_axis,history.history['val_accuracy'], color='orange', label='validation')
    plt.xlim(1,epo)
    plt.xticks(range(1,epo+1))
    plt.legend()
   
    
   
history = run_experiment()
result_plots(history) 
