import logging
import os

import tensorflow as tf
import keras
from keras import losses, optimizers

import args
from args import flags, FLAGS
from data import get_eval_set, get_test_set
from net import RBPN
from train import psnr

flags.DEFINE_string('model', None, 'Saved model for evaluation')
flags.DEFINE_string('input_dir', None, 'Low resolution images')
flags.DEFINE_string('label_dir', None, 'High resolution images')

logging.basicConfig(level=logging.INFO)

def main(not_parsed_args):
    logging.info('Loading evaluation dataset...')
    eval_dataset = get_test_set(FLAGS.input_dir, FLAGS.label_dir, FLAGS.frames, 1, 'filelist.txt', True, FLAGS.future_frame)
    logging.info('done. size %d' % len(eval_dataset))

    logging.info('Loading model...')
    model = RBPN()
    model.load_weights(FLAGS.model)
    model.compile(optimizer=optimizers.Adam(0.0001),
                loss=losses.mse,
                metrics=[psnr])

    logging.info('Evaluation start...')
    loss_total = 0
    psnr_total = 0
    for s in range(len(eval_dataset)):
        x, y = eval_dataset.batch()
        log = model.test_on_batch(x, y)
        logging.info('Step %d, loss %f psnr %f' % (s, log[0], log[1]))
        loss_total += log[0]
        psnr_total += log[1]
    loss_total /= len(eval_dataset)
    psnr_total /= len(eval_dataset)
    logging.info('Evaluation finished. Average loss %f psnr %f' % (loss_total, psnr_total))

if __name__ == '__main__':
    tf.app.run()