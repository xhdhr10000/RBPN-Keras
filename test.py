import sys
import os
import logging
import numpy as np
import tensorflow as tf
import keras
from keras.models import load_model
from PIL import Image

from data import get_eval_set, get_test_set
from train import loss, psnr
from net import RBPN
import args
from args import flags, FLAGS

flags.DEFINE_string('model', None, 'Saved model for evaluation')
flags.DEFINE_string('input_dir', None, 'Low resolution images')
flags.DEFINE_string('output_dir', None, 'Output high resolution images')

def cal_psnr(y_true, y_pred):
    print(y_true.shape)
    print(y_pred.shape)
    return 10. * np.log10(255.0*255.0 / np.mean(np.square(y_pred - y_true)))

def main(not_parsed_args):
    logging.info('Loading evaluation dataset...')
    test_dataset = get_test_set(FLAGS.input_dir, None, FLAGS.frames, 1, 'filelist.txt', True, FLAGS.future_frame)
    logging.info('done. size %d' % len(test_dataset))

    logging.info('Loading model...')
    model = RBPN()
    model.load_weights(FLAGS.model)
    # model.compile(optimizer=optimizers.Adam(FLAGS.lr),
    #             loss=losses.mae,
    #             metrics=[psnr])

    logging.info('Test start...')
    for s in range(len(test_dataset)):
        x, _, filenames = test_dataset.batch()
        y = model.predict(x)
        y *= 255.0
        Image.fromarray(y[0].astype(np.uint8), mode='rgb').save(os.path.join(FLAGS.output_dir, filenames[0]))
    logging.info('Test finished. Average loss %f psnr %f' % (loss_total, psnr_total))

    Image.fromarray(y[0,:,:,0].astype(np.uint8), mode='L').show()
    Image.fromarray(y_true[:,:,0].astype(np.uint8), mode='L').show()

if __name__ == '__main__':
    tf.app.run()