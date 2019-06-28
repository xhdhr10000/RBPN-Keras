import sys
import os
import logging
import numpy as np
import tensorflow as tf
import keras
from keras.models import load_model
from PIL import Image

from data import get_eval_set, get_test_set
from net import RBPN
import args
from args import flags, FLAGS

flags.DEFINE_string('model', None, 'Saved model for evaluation')
flags.DEFINE_string('input_dir', None, 'Low resolution images')
flags.DEFINE_string('output_dir', None, 'Output high resolution images')

logging.basicConfig(level=logging.INFO)

def main(not_parsed_args):
    logging.info('Loading evaluation dataset...')
    test_dataset = get_test_set(FLAGS.input_dir, None, FLAGS.frames, 4, 'filelist.txt', True, FLAGS.future_frame)
    logging.info('done. size %d' % len(test_dataset))

    logging.info('Loading model...')
    model = RBPN()
    model.load_weights(FLAGS.model)

    logging.info('Test start...')
    for s in range(len(test_dataset)):
        logging.info('Step %d' % s)
        x, _, filenames = test_dataset.batch(withName=True)
        y = model.predict(x)
        y *= 255.0
        Image.fromarray(y[0].astype(np.uint8), mode='RGB').save(os.path.join(FLAGS.output_dir, filenames[0]))
    logging.info('Test finished')

if __name__ == '__main__':
    tf.app.run()
