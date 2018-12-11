import numpy as np
import tensorflow as tf
from . import comparator_network as compnet
from . import embeding as emb

def make_ECM(new_frame,episodic_memory_buffer,par):
    '''

    This constructs the episodic curiosity module output tensors for a single frame to be compared against the given memory buffer. This function expects new_frame to be a single frame

    new_frame: [w,h,c]
    episodic_memory_buffer: [nbuf,nemb]

    '''
    
    new_emb = emb.embedding(new_frame,par,single = True)
    
    comp = tf.map_fn(lambda x:compnet.get_comparator(new_emb,x,single = True),episodic_memory_buffer)#[nbuf,1]

    return comp,new_emb

class episodic_curiosity_module:

    def __init__(self,frame_variable,adam,A = 1.,B = .5,threshold = .9,emb_size = 512,nbatch = 12):
        #buffer for memory
        self.buff_placeholder = tf.placeholder(tf.float32,shape = [None,emb_size])
        self.buff = []
        self.bufftime = []
        self.emb_size = emb_size
        self.threshold = .5
        #comparator input/output
        self.nbatch = nbatch
        
        self.frame1 = tf.placeholder(tf.float32,[self.nbatch,emb_size])
        self.frame2 = tf.placeholder(tf.float32,[self.nbatch,emb_size])
        
        self.complabel = tf.placeholder(tf.float32,[self.nbatch])
        
        self.comp_net_out = compnet.get_comparator(self.frame1,self.frame2,single = False)
        #comparator training function
        self.comploss = tf.pow(self.comp_net_out - tf.expand_dims(self.complabel,-1),2)
        
        self.comp_train = adam.minimize(tf.reduce_mean(self.comploss))

        #ECM operation
        self.ECM_new_input = frame_variable
        self.ECM = make_ECM(frame_variable,self.buff_placeholder,{"emb_size":emb_size})
        self.frame_embedding = emb.embedding(self.ECM_new_input,{"emb_size":emb_size},single = True)
            
    def run_ECM_on_video(self,frames,sess):

        out = [self.run_ECM(f,sess) for f in frames]

        return out

    def clean_buffer(self,cutoff = 1000):
        for k in reversed(range(len(self.bufftime))):
            if self.bufftime[k] > cutoff:
                self.remove_memory(k)
                
    def add_memory(self,enc):
        self.buff.append(enc)
        self.bufftime.append(0)

    def remove_memory(self,k):
        self.buff = self.buff[:k] + self.buff[k+1:]
        self.bufftime = self.bufftime[:k] + self.bufftime[k+1:]

    def run_ECM(self,frame,sess):

        if len(self.buff) == 0:
            out,emb = sess.run(self.ECM,{self.ECM_new_input:frame,self.buff_placeholder: np.zeros([1,self.emb_size])})
            self.add_memory(emb)

            return 1.

        else:
            out,emb = sess.run(self.ECM,{self.ECM_new_input:frame,self.buff_placeholder: np.array(self.buff)})

            percent = np.percentile(out,.1)
        
            for k in range(len(self.buff)):
                self.bufftime[k] += 1
                
            if percent > self.threshold:
                self.add_memory(emb)
        
            return (1 if percent > self.threshold else 0)

    
