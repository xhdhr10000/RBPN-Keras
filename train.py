import logging
import os

import numpy as np
from PIL import Image
import tensorflow as tf
import keras
from keras import losses, optimizers
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard

from data import get_training_set, get_eval_set
from net import SRModel, RBPN
import args

FLAGS = args.get()

def psnr(y_true, y_pred):
    return 10. * K.log(255.*255. / K.mean(K.square(y_pred - y_true))) / K.log(10.)

def loss(y_true, y_pred):
    p = psnr(y_true, y_pred)
    return 1/p

def load_weights(model):
    epoch = 0
    step = 0
    path_info = os.path.join(FLAGS.model_dir, 'info')
    if os.path.isfile(path_info):
        f = open(path_info)
        filename = f.readline()
        f.close()
        path = os.path.join(FLAGS.model_dir, filename)
        if os.path.isfile(path):
            logging.info('Loading %s' % filename)
            model.load_weights(path)
            epoch = int(filename.split('_')[1])
            step = int(filename.split('.')[0].split('_')[2])
    return epoch, step

def named_logs(model, logs, step):
  result = { 'batch': step, 'size': FLAGS.batch_size }
  for l in zip(model.metrics_names, logs):
    result[l[0]] = l[1]
  return result

def main(not_parsed_args):
    logging.info('Build dataset')
    # x_train, x_train_b, y_train, x_val, x_val_b, y_val = load_images()
    train_set = get_training_set(FLAGS.dataset, FLAGS.frames, FLAGS.scale, True, 'filelist.txt', True, FLAGS.patch_size, FLAGS.future_frame)
    if FLAGS.dataset_val:
        val_set = get_eval_set(FLAGS.dataset_val, FLAGS.frames, FLAGS.scale, True, 'filelist.txt', True, FLAGS.patch_size, FLAGS.future_frame)

    logging.info('Build model')
    model = RBPN()
    model.summary()
    last_epoch, last_step = load_weights(model)
    model.compile(optimizer=optimizers.Adam(0.0001),
                loss=losses.mse,
                metrics=[psnr])

    # checkpoint = ModelCheckpoint('models/model.hdf5', verbose=1)
    tensorboard = TensorBoard(log_dir='./tf_logs', batch_size=FLAGS.batch_size, write_graph=False, write_grads=True, write_images=True, update_freq='batch')
    tensorboard.set_model(model)

    logging.info('Training start')
    for e in range(last_epoch, FLAGS.epochs):
        tensorboard.on_epoch_begin(e)
        for s in range(last_step+1, len(train_set)):
            tensorboard.on_batch_begin(s)
            x, y = train_set.batch(FLAGS.batch_size)
            loss = model.train_on_batch(x, y)
            print('Epoch %d step %d, loss %f psnr %f' % (e, s, loss[0], loss[1]))
            tensorboard.on_batch_end(s, named_logs(model, loss, s))

            if FLAGS.dataset_val and s > 0 and s % FLAGS.val_interval == 0:
                logging.info('Validation start')
                val_loss = 0
                val_psnr = 0
                for j in range(len(val_set)):
                    x_val, y_val = val_set.batch(1)
                    score = model.test_on_batch(x_val, y_val)
                    val_loss += score[0]
                    val_psnr += score[1]
                val_loss /= len(val_set)
                val_psnr /= len(val_set)
                logging.info('Validation average loss %f psnr %f' % (val_loss, val_psnr))

            if s > 0 and s % FLAGS.save_interval == 0:
                logging.info('Saving model')
                filename = 'model_%d_%d.h5' % (e+1, s)
                path = os.path.join(FLAGS.model_dir, filename)
                path_info = os.path.join(FLAGS.model_dir, 'info')
                model.save_weights(path)
                f = open(path_info, 'w')
                f.write(filename)
                f.close()
        tensorboard.on_epoch_end(e)

if __name__ == '__main__':
    tf.app.run()
