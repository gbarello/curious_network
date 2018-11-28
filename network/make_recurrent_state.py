import tensorflow as tf

def make_recurrent_state(image,initial_rec_state,actions,enc_layers,ff_layers,rec_layers,action_layers,ain_layers,pred_layers):
    
    enc = []
    rec = []
    pred = []
    act = []
    
    recurrent_state = initial_rec_state
    
    for t in range(int(image.shape[1])):
        encoding,recurrent_state,prediction,action = get_next_state(image[:,t],
                                                                    recurrent_state,
                                                                    actions[:,t],
                                                                    enc_layers,
                                                                    ff_layers,
                                                                    rec_layers,
                                                                    action_layers,
                                                                    ain_layers,
                                                                    pred_layers)
        
        enc.append(encoding)
        rec.append(recurrent_state)
        pred.append(prediction)
        act.append(action)
        
    return {"rec_state":tf.stack(rec,1),"encoding":tf.stack(enc,1),"actions":tf.stack(act,1),"prediction":tf.stack(pred,1)}

def get_next_state(image_in,rec_state,action,enc_layers,ff_layers,rec_layers,action_layers,ain_layers,pred_layers):
    image = tf.reshape(image_in,[image_in.shape[0],-1])

    encoding = make_encoder(image,enc_layers,name = "encoder")
    
#    ff_input = make_encoder(encoding,ff_layers + [rec_state.shape[-1]],name = "ff_in")
#    rec_input = make_encoder(rec_state,rec_layers + [rec_state.shape[-1]],name = "rec_in")
#    action_input = make_encoder(action,ain_layers + [rec_state.shape[-1]],name = "act_in")

    rec_state_inp = tf.concat([encoding,rec_state,action],axis = -1)
    new_rec_state = make_encoder(rec_state_inp,rec_layers + [rec_state.shape[-1]],name = "rec_in",nonlin = tf.nn.relu)
    
#    new_rec_state = tf.nn.sigmoid(ff_input + rec_input + action_input)
    
    prediction = make_encoder(tf.stop_gradient(new_rec_state),pred_layers + [encoding.shape[-1]],name = "prediction")
    next_action = make_encoder(tf.stop_gradient(new_rec_state),action_layers + [action.shape[-1]],name = "action")
    next_action = tf.nn.softmax(next_action)
    
    return encoding, new_rec_state,prediction, next_action
    

def make_encoder(input_tensor,layers,name,reuse = tf.AUTO_REUSE,nonlin = tf.nn.relu):
    net = input_tensor

    for l in range(len(layers)):
        net = tf.layers.dense(net,layers[l],activation = nonlin,name = name + "_{}".format(l),reuse = reuse)

    return net
