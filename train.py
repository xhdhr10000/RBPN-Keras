import numpy as np
from PIL import Image
import keras
from keras import losses, optimizers
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard

from data import get_training_set
from net import SRModel, RBPN

def psnr(y_true, y_pred):
    return 10. * K.log(255.*255. / K.mean(K.square(y_pred - y_true))) / K.log(10.)

def loss(y_true, y_pred):
    p = psnr(y_true, y_pred)
    return 1/p

def train():
    # x_train, x_train_b, y_train, x_val, x_val_b, y_val = load_images()
    train_set = get_training_set('dataset/image_h/train', 7, 4, True, 'filelist.txt', True, 64, True)

    model = RBPN()
    model.summary()
    model.compile(optimizer=optimizers.Adam(0.0002),
                loss=losses.mse,
                metrics=[psnr])

    checkpoint = ModelCheckpoint('models/model.hdf5', verbose=1)
    tensorboard = TensorBoard(log_dir='./tf_logs')

    for i in range(len(train_set)):
        x, y = train_set.batch(4)
        loss = model.train_on_batch(x, y)
        print('Step %d, loss %f, psnr %f' % (i, loss[0], loss[1]))

        # score = model.evaluate([x_val, x_val_b], y_val)
        # print('Test loss: ', score[0])
        # print('Test acc: ', score[1])

if __name__ == '__main__':
    train()
