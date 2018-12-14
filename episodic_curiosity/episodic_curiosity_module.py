import numpy as np
import tensorflow as tf
from . import comparator_network as compnet
from . import embeding as emb

class memory:
    def __init__(self,emb, emb_size, mem_limit):
        assert emb_size == int(emb.shape[0])

        self.emb = emb
        
        self.mem_limit = tf.constant(mem_limit,dtype = tf.int32)
        self.buff = tf.Variable(np.zeros([mem_limit,emb_size],dtype = np.float32))
        self.nmem = tf.Variable(np.int32(0))

        self.add_memory_op = self.make_add_mem_op(self.emb)
        self.clear_memory_op = self.make_clear_buffer_op()

        self.add_memory = lambda mem,sess:sess.run(self.add_memory_op,{self.emb:mem})
        self.clear_memory = lambda sess:sess.run(self.clear_memory_op)

    def memories(self):
        return self.buff[:self.nmem]

    def make_add_mem_op(self,memory):
        
        def add_mem_to_end():
            newmem = tf.assign(self.buff[self.nmem],memory)
            with tf.control_dependencies([newmem]):
                upnmem = tf.assign(self.nmem,self.nmem + 1)
            return upnmem - 1, newmem
        
        def add_mem_to_rand():
            rand = tf.random_uniform(self.nmem.shape,0,self.nmem,dtype = tf.int32)
            newmem = tf.assign(self.buff[rand],memory)
            return rand, newmem

        return tf.cond(self.nmem < self.mem_limit,add_mem_to_end,add_mem_to_rand)

    def make_clear_buffer_op(self):
        nmemclear = tf.assign(self.nmem,tf.zeros_like(self.nmem,dtype = tf.int32))
        memclear = tf.assign(self.buff,tf.zeros_like(self.buff,dtype = tf.float32))
        return memclear,nmemclear

class novelty_bonus_module:
    '''
    Description: Build a novelty bonus module which stores memories, and returns novelty scores of new memories
    '''
    def __init__(self,embedding,episode,comparator_func,memmax,threshold = .75,score_fn = lambda x:tf.reduce_mean(x)):
        '''
        Description: initializer function which builds the memory module and all the tensros for scoring new memories and computing the novelty bonus.
        
        args:
        embedding - a 1-D Tensor variable which represents embeddings of inputs. Shape [embedding_size].
        comparator_func - a function which takes two Tensor variables and returns a scalar output tensor representing their probability of reachability.
        memmax - (int) the maximum number of memories to retain.
        storage_threshold - (float) the threshold of score with which to retain a new memory. default is .75.
        score_fn - a function which computes the score from a list of caparator output values. Default is the min function.

        Note: In contrast to https://arxiv.org/abs/1810.02274 the comparator output here is the probability of novelty (in theirs it was the probability of non-novelty), hence usng the max instead of the min for scoring.
        
        '''
        assert embedding.shape[0] == episode.shape[1]
        
        self.embedding = embedding
        self.episode = episode
        self.comparator_fn = comparator_func
        self.memory = memory(embedding,embedding.shape[0],memmax)
        
        self.storage_threshold = threshold
        self.score_fn = score_fn

        self.score_episode_op = tf.map_fn(lambda x:self._build_score_memory_op(x),self.episode,dtype = (tf.float32,tf.float32,(tf.int32,tf.float32)))
        
        self.score_memory = lambda emb,sess:sess.run(self._build_score_memory_op(self.embedding),{self.embedding:emb})
        self.score_episode = lambda emb,sess:sess.run(tf.map_fn(lambda x:self._build_score_memory_op(x),self.episode,dtype = (tf.float32,tf.float32,(tf.int32,tf.float32))),{self.episode:emb})
        
    def _build_novelty_variable(self,emb):
        return tf.map_fn(lambda x:self.comparator_fn(emb,x,single = True),self.memory.memories())
    
    def _build_score_memory_op(self,emb):
        nomems = tf.logical_not(self.memory.nmem > 0)
        
        novelty = tf.cond(nomems,lambda:np.array([0],dtype = np.float32),lambda:self._build_novelty_variable(emb))
        score = self.score_fn(novelty)
        highscore = (score > self.storage_threshold)
        add_mem_func = lambda:self.memory.make_add_mem_op(emb)
        
        condout = tf.cond(tf.logical_or(nomems,highscore),add_mem_func,lambda:(self.memory.nmem,self.memory.buff))
        
        return novelty,score,condout
        
class episodic_curiosity_module:
    def __init__(self,frame_variable,adam,A = 1.,B = .5,threshold = .5,emb_size = 64,nbatch = 12,dense = False):
        #buffer for memory
        self.buff_placeholder = tf.placeholder(tf.float32,shape = [None,emb_size])
        self.buff = []
        self.bufftime = []
        self.emb_size = emb_size
        self.threshold = threshold
        #comparator input/output
        self.nbatch = nbatch
        
        self.frame1 = tf.placeholder(tf.float32,[self.nbatch,emb_size])
        self.frame2 = tf.placeholder(tf.float32,[self.nbatch,emb_size])
        
        self.complabel = tf.placeholder(tf.float32,[self.nbatch])

        self.comp_net_out = self.get_comparator()
        
        #comparator training function
        self.comploss = -tf.log(.01 + .99*self.comp_net_out)*tf.expand_dims(self.complabel,-1) - tf.log(.01 + 1 - .99*self.comp_net_out)*tf.expand_dims(1 - self.complabel,-1)
        
        self.comp_train = adam.minimize(tf.reduce_mean(self.comploss))

        #ECM operation
        self.ECM_new_input = frame_variable
        self.ECM = self.make_ECM()
        self.frame_embedding = self.get_embedding(single = True)

    def make_ECM(self):
        '''
        
        This constructs the episodic curiosity module output tensors for a single frame to be compared against the given memory buffer. This function expects new_frame to be a single frame

        new_frame: [w,h,c]
        episodic_memory_buffer: [nbuf,nemb]
        
        '''
        episodic_memory_buffer = self.buff_placeholder
        new_emb = self.get_embedding(single = True)
        
        comp = tf.map_fn(lambda x:compnet.get_comparator(new_emb,x,single = True),episodic_memory_buffer)#[nbuf,1]
        
        return comp,new_emb
        
    def run_ECM_on_video(self,frame,sess):
        '''
        Takes a sequence of observations `frame` and a tf session `sess` and returns a list of intrinsic rewards.
        '''
        out = [self.run_ECM(f,sess) for f in frame]

        return out

    def run_ECM(self,frame,sess):

        if len(self.buff) == 0:
            emb = sess.run(self.frame_embedding,{self.ECM_new_input:frame})
            self.add_memory(emb)

            return 0.

        else:
            out,emb = sess.run(self.ECM,{self.ECM_new_input:frame,self.buff_placeholder: np.array(self.buff)})
        
            for k in range(len(self.buff)):
                self.bufftime[k] += 1
                
            percent = np.percentile(out,.1)
            
            if percent > self.threshold:
                self.add_memory(emb)
        
            return (percent/self.threshold)**2

    def clean_buffer(self,cutoff = 0):
        for k in reversed(range(len(self.bufftime))):
            if self.bufftime[k] > cutoff:
                self.remove_memory(k)
                
    def add_memory(self,enc):
        self.buff.append(enc)
        self.bufftime.append(0)

    def remove_memory(self,k):
        self.buff = self.buff[:k] + self.buff[k+1:]
        self.bufftime = self.bufftime[:k] + self.bufftime[k+1:]
    
    def get_comparator(self,single = False):
        return compnet.get_comparator(self.frame1,self.frame2,single = False)

    def get_embedding(self,single = False):
        return NotImplementedError

    
class CNN_ECM(episodic_curiosity_module):

    def __init__(self,frame_variable,adam,**kwargs):#A = 1.,B = .5,threshold = .9,emb_size = 64,nbatch = 12):
        episodic_curiosity_module.__init__(self,frame_variable,adam,**kwargs)#A,B,threshold,emb_size,nbatch)

    def get_embedding(self,single = False):
        return emb.embedding(self.ECM_new_input,{"emb_size":self.emb_size},single = True)    

class DENSE_ECM(episodic_curiosity_module):

    def __init__(self,frame_variable,adam,**kwargs):#A = 1.,B = .5,threshold = .9,emb_size = 64,nbatch = 12):
        episodic_curiosity_module.__init__(self,frame_variable,adam,**kwargs)#A,B,threshold,emb_size,nbatch)

    def get_embedding(self,single = False):
        return emb.dense_embedding(self.ECM_new_input,{"emb_size":self.emb_size},single = True)


