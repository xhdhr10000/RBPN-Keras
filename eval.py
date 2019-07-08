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
flags.DEFINE_string('model2', None, 'Another model for average')
flags.DEFINE_string('input_dir', None, 'Low resolution images')
flags.DEFINE_string('label_dir', None, 'High resolution images')

logging.basicConfig(level=logging.INFO)

def cal_psnr(y_true, y_pred):
    return 10. * np.log10(255.0*255.0 / np.mean(np.square(y_pred*255. - y_true*255.)))

def main(not_parsed_args):
    logging.info('Loading evaluation dataset...')
    eval_dataset = get_test_set(FLAGS.input_dir, FLAGS.label_dir, FLAGS.frames, 4, 'filelist.txt', True, FLAGS.future_frame)
    logging.info('done. size %d' % len(eval_dataset))

    logging.info('Loading model...')
    model = RBPN()
    model.load_weights(FLAGS.model)
    model.compile(optimizer=optimizers.Adam(FLAGS.lr),
                loss=losses.mae,
                metrics=[psnr])
    if FLAGS.model2:
        model2 = RBPN()
        model2.load_weights(FLAGS.model2)
        model2.compile(optimizer=optimizers.Adam(FLAGS.lr),
                loss=losses.mae,
                metrics=[psnr])

    logging.info('Evaluation start...')
    loss_total = 0
    psnr_total = 0
    for s in range(len(eval_dataset)):
        x, y, _ = eval_dataset.batch()
        if FLAGS.model2:
            y1 = model.predict(x)
            y2 = model2.predict(x)
            log = cal_psnr(y, (y1+y2)/2.0)
            logging.info('Step %d, psnr %f' % (s, log))
            psnr_total += log
        else:
            log = model.test_on_batch(x, y)
            logging.info('Step %d, loss %f psnr %f' % (s, log[0], log[1]))
            loss_total += log[0]
            psnr_total += log[1]
    loss_total /= len(eval_dataset)
    psnr_total /= len(eval_dataset)
    logging.info('Evaluation finished. Average loss %f psnr %f' % (loss_total, psnr_total))

if __name__ == '__main__':
    tf.app.run()
