import tensorflow as tf
import sys
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('scale', 4, 'Scale factor for super resolution')
flags.DEFINE_integer('patch_size', 32, 'Patch size for training')
flags.DEFINE_integer('frames', 7, 'Frames to use for one-frame super resolution training')
flags.DEFINE_boolean('future_frame', True, 'Use future frame for training')
flags.DEFINE_boolean('residual', True, 'Train with bicubic image')
flags.DEFINE_boolean('bn', False, 'Batch normalization after feat0 & feat1')

flags.DEFINE_integer('feat_filters', 256, 'Number of filters to extract feature')
flags.DEFINE_integer('res_number', 5, 'Number of resnet blocks in feature extractor')

flags.DEFINE_integer('epochs', 10000, 'Training epochs')
flags.DEFINE_integer('batch_size', 2, 'Batch size for training')
flags.DEFINE_float('lr', 0.0001, 'Learning rate')
flags.DEFINE_string('dataset_h', 'dataset', 'Dataset of hr images for training')
flags.DEFINE_string('dataset_l', None, 'Dataset of lr images for training, if not defined, scale down from hr')
flags.DEFINE_string('dataset_val_h', None, 'Dataset for validation')
flags.DEFINE_string('dataset_val_l', None, 'Dataset for validation')
flags.DEFINE_string('model_dir', 'models', 'Model save directory')
flags.DEFINE_integer('val_interval', 200, 'Validation interval')
flags.DEFINE_integer('save_interval', 200, 'Model save interval')

def get():
    print("Python Interpreter version:%s" % sys.version[:3])
    print("tensorflow version:%s" % tf.__version__)
    print("numpy version:%s" % np.__version__)

    # check which library you are using
    # np.show_config()
    return FLAGS