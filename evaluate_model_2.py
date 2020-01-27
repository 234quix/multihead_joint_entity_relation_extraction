import utils
import tf_utils
from build_data import build_data
import numpy as np
import tensorflow as tf
import sys
import os.path


configpath = './configs/CoNLL04/bio_config_adv'
print('loading config')
config=build_data(configpath)


print('loading dev data')
dev_data = utils.HeadData(config.dev_id_docs, np.arange(len(config.dev_id_docs)))



# Step 1 load the graph
with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('my_test_model.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./'))

    # try to get the names of the operations: 
    for op in tf.get_default_graph().get_operations():
        print(str(op.name))
