from keras.datasets import mnist
from keras import utils
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape + (1,))
x_test = x_test.reshape(x_test.shape + (1,))

y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)

print('Data shape: x_train:{} y_train:{} x_test:{} y_test:{}'.format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))