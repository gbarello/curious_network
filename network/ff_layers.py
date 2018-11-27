import tensorflow as tf

def make_encoder(input_tensor,layers,name,reuse = False):
    net = input_tensor

    for l in range(len(layers)):
        net = tf.layers.Dense(net,layers[l],name = name + "_{}".format(l),reuse = reuse)

    return net
