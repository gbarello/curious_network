import tensorflow as tf
from . import building_blocks as blk

def get_comparator(I1,I2,single = False):

    if single:
        inputs = tf.expand_dims(tf.concat([I1,I2],0),0)
    else:
        inputs = tf.concat([I1,I2],axis = 1)
    novel = blk.make_dense_encoder(inputs,[128,64,32,1],"compnet",outnonlin = tf.nn.sigmoid)

    if single:
        out = novel[0]
    else:
        out = novel
    
    return novel

