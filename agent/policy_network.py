import numpy as np
import tensorflow as tf

def make_policy_net(inputs,n_output,dense_layers,cnn_layers,size,stride,pool,poolstride,single = False):

    if single:
        net = tf.expand_dims(inputs,0)
    else:
        net = inputs
        
    for k in range(len(cnn_layers)):
        net = tf.layers.conv2d(net,cnn_layers[k],size[k],stride[k],activation = tf.nn.relu)
        if pool[k] != 0:
            net = tf.layers.max_pooling2d(net,pool[k],poolstride[k])

    nout = [int(i) for i in net.shape[1:]]
    net = tf.reshape(net,[-1,np.prod(nout)])
    
    for k in range(len(dense_layers)):
        net = tf.layers.dense(net,dense_layers[k],activation = tf.nn.relu)
        net = tf.layers.dropout(net,.5)

    net = tf.layers.dense(net,n_output,activation = lambda x:x)
    net = tf.nn.softmax(net)
    if single:
        net = net[0]
    
    return net
