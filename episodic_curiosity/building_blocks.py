import tensorflow as tf

def make_dense_encoder(input_tensor,layers,name,reuse = tf.AUTO_REUSE,nonlin = tf.nn.relu,outnonlin = lambda x:x):
    net = tf.reshape(input_tensor,[int(input_tensor.shape[0]),-1])

    for l in range(len(layers)-1):
        net = tf.layers.dense(net,layers[l],activation = nonlin,name = name + "_{}".format(l),reuse = reuse)
        
    net = tf.layers.dense(net,layers[-1],activation = outnonlin,name = name + "_{}".format(l + 1),reuse = reuse)
        
    return net
    

def make_cnn_encoder(input_tensor,layers,name,reuse = tf.AUTO_REUSE,nonlin = tf.nn.relu,outnonlin = lambda x:x):
    net = input_tensor
    
    for l in range(len(layers)-1):
        net = tf.layers.conv2d(net,layers[l],3,activation = nonlin,padding = "same",name = name + "_{}".format(l),reuse = reuse)
        net = tf.layers.max_pooling2d(net,2,2)
        
    net = tf.layers.conv2d(net,layers[l],3,activation = outnonlin,padding = "same",name = name + "_{}".format(l+1),reuse = reuse)
        
    return net
