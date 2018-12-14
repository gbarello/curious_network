import os
import numpy as np
from . import episodic_curiosity_module as epc
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def test_memory_module():
    tf.reset_default_graph()
    esize = 4
    maxmem = 3
    embvar = tf.placeholder(tf.float32,[esize])
    
    mem = epc.memory(embvar,esize,maxmem)
        
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    memsize = sess.run(mem.buff).shape
    #make sure memory buffer is the right size
    assert memsize == (maxmem,esize)
    
    for k in range(int(1.5*esize)):
        #verify there are the rigth nimber of memories
        assert sess.run(mem.nmem) == min(k,maxmem)

        temp = np.random.randn(esize)
        loc, buff = mem.add_memory(temp,sess)

        #verify the right memory went in the right spot        
        assert np.allclose(buff[loc], temp)
        
    mem.clear_memory(sess)
    buff,nmem = sess.run([mem.buff,mem.nmem])
    #verify the buffer was cleared
    
    assert np.all(buff == np.zeros_like(buff))
    #verify nmem was reset
    assert nmem == 0
    print("Memory module test passed.")
    sess.close()

def test_novelty_bonus_module():
    tf.reset_default_graph()

    esize = 2
    maxmem = 10
    
    emb = tf.placeholder(tf.float32,[esize],name = "embedding")
    eps = tf.placeholder(tf.float32,[None,esize],name = "episode")

    def comp(e1,e2,single = True):
        return tf.reduce_sum(tf.pow(e1-e2,2))
    
    NBM = epc.novelty_bonus_module(emb,eps,comp,maxmem)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    rand = np.random.randn(100,esize)

    for k in range(5):
        res = NBM.score_memory(rand[k],sess)
        print(res[0])
        
    res = sess.run(NBM.score_episode_op,{eps:rand[-2:]})
        
    print(res[2][0])
    print(res[2][1])

    print(res[1])

    sess.close()
    
def main():
    test_memory_module()
    test_novelty_bonus_module()
