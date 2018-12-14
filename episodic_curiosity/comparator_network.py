import tensorflow as tf
from . import building_blocks as blk

def get_comparator(I1,I2,single = False):

    a = I1 - I2
    b = I1 + I2
    
    if single:
        inputs = tf.expand_dims(tf.concat([a,b],0),0)
    else:
        inputs = tf.concat([a,b],axis = 1)


    
    novel = blk.make_dense_encoder(inputs,[32,32,1],"compnet",outnonlin = tf.nn.sigmoid)

    if single:
        out = novel[0]
    else:
        out = novel
    
    return novel

