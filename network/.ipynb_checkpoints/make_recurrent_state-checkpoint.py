import tensorflow as tf

def make_recurrent_state(ff_state,initial_state,n_outputs,fflayers,reclayers,extlayers,extinlayers,recnl):

    out = []
    actions = []
    
    recurrent_state = initial_state
    
    for t in range(int(ff_state.shape[1])):
        action = make_encoder(tf.stop_gradient(recurrent_state),extlayers + [n_outputs],"rec_to_ext",reuse = tf.AUTO_REUSE,nonlin = tf.nn.sigmoid)
        action = tf.nn.softmax(action)
        
        acut = tf.stop_gradient(action)
        action_input = make_encoder(acut,extinlayers + [initial_state.shape[-1]],"ext_to_rec")
       
        ff_input = make_encoder(ff_state[:,t],fflayers + [initial_state.shape[-1]],"ff_to_rec",reuse = tf.AUTO_REUSE)
        recurrent_input = make_encoder(recurrent_state,reclayers + [initial_state.shape[-1]],"rec_to_rec",reuse = tf.AUTO_REUSE)

        recurrent_state = recnl(ff_input + recurrent_input + action_input)
        out.append(recurrent_state)
        actions.append(action)
        
    return {"rec_state":tf.stack(out,1),"actions":tf.stack(actions,1)}

def make_encoder(input_tensor,layers,name,reuse = tf.AUTO_REUSE,nonlin = tf.nn.relu):
    net = input_tensor

    for l in range(len(layers)):
        net = tf.layers.dense(net,layers[l],activation = nonlin,name = name + "_{}".format(l),reuse = reuse)

    return net
