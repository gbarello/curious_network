import tensorflow as tf
from . import building_blocks as blk

def embedding(I,param,single = False):
    if single:
        inputs = tf.expand_dims(I,0)
    else:
        inputs = I
        
    emb = blk.make_cnn_encoder(inputs,[32,64,128],"cnn_emb",outnonlin = tf.nn.relu)
    emb = blk.make_dense_encoder(emb,[256,param["emb_size"]],"den_emb",outnonlin = tf.nn.relu)

    print(emb,single)
    if single:
        emb = emb[0]

    return emb

def dense_embedding(I,param,single = False):
    if single:
        inputs = tf.expand_dims(I,0)
    else:
        inputs = I
        
    emb = blk.make_dense_encoder(inputs,[param["emb_size"]],"den_emb",outnonlin = lambda x:x)

    if single:
        emb = emb[0]

    return emb
