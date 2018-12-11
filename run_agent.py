import numpy as np
import tensorflow as tf
import gym
import numpy as np
import episodic_curiosity.episodic_curiosity_module as ECM_build
import agent.policy_network as policy
import os

os.CUDA_VISIBLE_DEVICES = "2"

np.random.seed(1)
tf.set_random_seed(0)

def make_system():

    env = gym.make("MontezumaRevenge-v0")

    o = env.reset()

    frame_input = tf.placeholder(tf.float32,o.shape)
    many_frame_input = tf.placeholder(tf.float32,[None]+ list(o.shape))
    adam = tf.train.AdamOptimizer()
    
    episodic_mem = ECM_build.episodic_curiosity_module(frame_input,adam)

    pol_net = policy.make_policy_net(frame_input,18,[512],[512,256,128,64],[3,3,3,3],[2,1,1,1],[2,2,2,2],[1,1,1,1],single = True)
    many_pol_net = policy.make_policy_net(many_frame_input,18,[512],[512,256,128,64],[3,3,3,3],[2,1,1,1],[2,2,2,2],[1,1,1,1],single = False)

    return {"env":env,"frame_input":frame_input,"optimizer":adam,"ECM":episodic_mem,"full_policy":many_pol_net,"policy":pol_net,"iframe":o,"many_frames":many_frame_input}

def make_cp_batch(img,n,cut = 3,kap = 2):

    close = [[img[a],img[b]] for a in range(len(img) - cut) for b in range(cut)]
    far = [[img[a],img[b]] for a in range(len(img) - kap*cut+1) for b in range(a + kap*cut,len(img))]

    label = np.random.randint(0,2,n)

    allbatch = [close,far]
    
    batch = [allbatch[k][np.random.choice(len(allbatch[k]))] for k in label]

    return np.array(batch),np.array(label)

def run_episode(eplen,agent_stuff,init_obs,sess,arep = 3,policy = True):
    ECM = agent_stuff["ECM"]
    o = init_obs
    
    a_dist,femb = sess.run([agent_stuff["policy"],ECM.frame_embedding],{agent_stuff["frame_input"]:agent_stuff["iframe"]})

    obs = []
    action = []
    pact = []
    erew = []
    embed = []
    
    for k in range(eplen):
        obs.append(o)
        embed.append(femb)
        if policy:
            act = np.random.choice(range(18),p = a_dist)
        else:
            act = np.random.choice(range(18))

        rew = 0

        done = False
        for step in range(arep):
            o,r,d,i = agent_stuff["env"].step(act)
            rew += r
            done = (done or d)
            
        pact.append(a_dist)
        action.append(act)
        erew.append(rew)

        a_dist,femb = sess.run([agent_stuff["policy"],ECM.frame_embedding],{agent_stuff["frame_input"]:o})

        if done:
            print(done)
            o = agent_stuff["env"].reset()

    return obs,action,pact,erew,embed

def main():

    agent_stuff = make_system()
    ECM = agent_stuff["ECM"]
    
    nepc = 1000
    eplen = 50

    alpha = 1
    beta = 0

    action_inp = tf.placeholder(tf.float32,[None,18])
    action_prob = tf.placeholder(tf.float32,[None,18])
    reward_inp = tf.placeholder(tf.float32,[None])

    policy_loss = tf.reduce_mean(tf.log(tf.reduce_sum(action_inp*agent_stuff["full_policy"],axis = 1))*reward_inp)

    policy_train = agent_stuff["optimizer"].minimize(-policy_loss)

    init_obs = agent_stuff["iframe"]

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    for epoch in range(nepc):
        print(epoch)
        obs,action,pact,erew,embed = run_episode(eplen,agent_stuff,init_obs,sess,policy = True if epoch > 100 else False)
        init_obs = obs[-1]
        
        cp_batch, cp_label = make_cp_batch(embed,ECM.nbatch)

        _,cout,cscore = sess.run([ECM.comp_train,ECM.comp_net_out,ECM.comploss],{ECM.frame1:cp_batch[:,0],ECM.frame2:cp_batch[:,1],ECM.complabel:cp_label})
        print(np.mean(cscore))
        
        if epoch > 1:
            cp_out = ECM.run_ECM_on_video(obs,sess)

            r_int = get_intrinsic_rew(cp_out,alpha,beta)

            action_1hot = np.zeros([len(action),18])
            action_1hot[np.arange(len(action)),action] = 1.
            
            policy_perf,_ = sess.run([policy_loss,policy_train],{action_inp : action_1hot,agent_stuff["many_frames"] : np.array(obs), reward_inp : r_int + np.array(erew)})
                       
            print("{}\t{}".format(policy_perf,len(ECM.buff)))
            ECM.clean_buffer(10)
            
def get_intrinsic_rew(comparator_output,a,b):
    return a * np.sum(np.array(comparator_output) - b)

if __name__ == "__main__":
    main()
        
