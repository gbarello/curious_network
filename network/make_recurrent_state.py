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

    rec_state_inp = tf.concat([encoding,.75*rec_state,action],axis = -1)
    new_rec_state = make_encoder(rec_state_inp,rec_layers + [rec_state.shape[-1]],name = "rec_in",nonlin = lambda x:x)
    
#    new_rec_state = tf.nn.sigmoid(ff_input + rec_input + action_input)
    
#    prediction = make_encoder(tf.stop_gradient(new_rec_state),pred_layers + [encoding.shape[-1]],name = "prediction")
    prediction = make_encoder(tf.concat([new_rec_state,action],axis = -1),pred_layers,name = "prediction",nonlin = tf.nn.relu)
    prediction = make_encoder(prediction,[image.shape[-1]],name = "prediction_out",nonlin = lambda x:x)

#    next_action = make_encoder(tf.concat([tf.stop_gradient(new_rec_state),image],axis = -1),action_layers + [action.shape[-1]],name = "action")
    next_action = make_encoder(tf.concat([image,action,tf.stop_gradient(new_rec_state)],axis = -1),action_layers + [action.shape[-1]],name = "action")
    next_action = tf.nn.softmax(next_action)
    
    return encoding, new_rec_state,prediction, next_action
    

def make_encoder(input_tensor,layers,name,reuse = tf.AUTO_REUSE,nonlin = tf.nn.relu):
    net = input_tensor

    for l in range(len(layers)):
        net = tf.layers.dense(net,layers[l],activation = nonlin,name = name + "_{}".format(l),reuse = reuse)

    return net

def get_pred_network_architechture(nb,view_size,time,rec_size,number_of_actions,enc_layers,ff_layers,rec_layers,action_layers,ain_layers,pred_layers):
    tf.reset_default_graph()

    rec_init_state = tf.placeholder(tf.float32,[nb,rec_size])

    im_in = tf.placeholder(tf.float32,[nb,time,view_size,view_size])
    act_in = tf.placeholder(tf.float32,[nb,time,number_of_actions])
    disc_rew = tf.placeholder(tf.float32,[nb,time-1])

    single_im_in = tf.placeholder(tf.float32,[nb,1,view_size,view_size])
    single_act_in = tf.placeholder(tf.float32,[nb,1,number_of_actions])

    single_network = make_recurrent_state(single_im_in,rec_init_state,single_act_in,enc_layers,ff_layers,rec_layers,action_layers,ain_layers,pred_layers)

    full_network = make_recurrent_state(im_in,rec_init_state,act_in,enc_layers,ff_layers,rec_layers,action_layers,ain_layers,pred_layers)
    
    
    
    return {"full":full_network,"single":single_network,"single_im":single_im_in,"single_act":single_act_in,"image":im_in,"action":act_in,"reward":disc_rew,"rec_init":rec_init_state}

def get_TP_network_architechture(single_in, many_in, single_act, many_act, enc1_layers, pred1_layers, enc2_layers, pred2_layers, action_layers):
    
    single_run = two_layer_prediction(single_in, single_act, enc1_layers, pred1_layers, enc2_layers, pred2_layers, action_layers)
    
    IN = tf.reshape(many_in,[-1,many_in.shape[-1]])
    ACT = tf.reshape(many_act,[-1,many_act.shape[-1]])

    many_run = two_layer_prediction(IN, ACT, enc1_layers, pred1_layers, enc2_layers, pred2_layers, action_layers)
    for d in many_run.keys():
        many_run[d] = tf.reshape(many_run[d],[many_in.shape[0],many_in.shape[1],-1])
        
    return single_run,many_run
    
def two_layer_prediction(input_tensor, action, enc1_layers, pred1_layers, enc2_layers, pred2_layers, action_layers):
    
    enc1 = make_encoder(tf.concat([input_tensor,action],axis = -1),enc1_layers,"enc1")
    pred1 = make_encoder(enc1,pred1_layers,"pred1")
    pred1 = make_encoder(pred1,[input_tensor.shape[-1]],"pred1_out",nonlin = lambda x:x)
    
    enc2 = make_encoder(tf.concat([tf.stop_gradient(enc1),action],axis = -1),enc2_layers,"enc2")
    pred2 = make_encoder(enc2,pred2_layers + [enc1.shape[-1]],"pred2")
    
    next_action = make_encoder(tf.concat([input_tensor,tf.stop_gradient(enc2)],axis = -1),action_layers + [action.shape[-1]],"action")
    next_action = tf.nn.softmax(next_action)
    
    return {"enc1":enc1,"enc2":enc2,"pred1":pred1,"pred2":pred2,"next_action":next_action}