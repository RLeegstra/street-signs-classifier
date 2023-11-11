import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, GlobalAvgPool2D

from deeplearning_models import functional_model, MyCustomModel



def display_examples(examples, labels):
    plt.figure(figsize=(10,10))

    for i in range(25):
        idx = np.random.randint(0, examples.shape[0]-1) # shape[n] is the n'th image. Pick random image between [0, last_image]
        img = examples[idx]
        label = labels[idx]


        plt.subplot(5, 5, i+1)
        plt.title(str(label))
        plt.tight_layout()
        plt.imshow(img, cmap = 'gray')
    
    plt.show()




if __name__=='__main__':
    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() # example dataset


    print("x_train.shape = ", x_train.shape)
    print("y_train.shape = ", y_train.shape)
    print("x_test.shape = ", x_test.shape)
    print("y_test.shape = ", y_test.shape)


    x_train = x_train.astype('float32') / 255  # normalize to 0 - 1 instead of 0 - 255 (not always required)
    x_test = x_test.astype('float32') / 255

    x_train = np.expand_dims(x_train, axis=-1) # model accepts input with shape 28x28x1 (see model), so add 1 dimension 
    x_test = np.expand_dims(x_test, axis=-1)

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10) # convert to one hot encoding for categorical_crossentropy loss function
    # label = 2 --> one hot encoding = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

    #model = functional_model()
    model = MyCustomModel()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')


    # train
    model.fit(x_train, y_train, batch_size=64, epochs=3, validation_split=0.2) # validation_split = use 20% of train set for validation

    # evaluation on test set
    model.evaluate(x_test, y_test, batch_size=64)
