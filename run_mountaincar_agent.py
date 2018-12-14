import numpy as np
import tensorflow as tf
import gym
import numpy as np
import episodic_curiosity.episodic_curiosity_module as ECM_build
import agent.policy_network as policy
import os

os.environ["CUDA_VISIBLE_DEVICES"]="2"

np.random.seed(1)
tf.set_random_seed(0)

def make_system():

    env = gym.make("MountainCar-v0")

    o = env.reset()

    frame_input = tf.placeholder(tf.float32,o.shape)
    many_frame_input = tf.placeholder(tf.float32,[None]+ list(o.shape))
    adam = tf.train.AdamOptimizer()
    
    episodic_mem = ECM_build.DENSE_ECM(frame_input,adam,emb_size = 2,threshold = .5)

    pol_net = policy.make_dense_policy_net(frame_input,3,[8,8],single = True)
    many_pol_net = policy.make_dense_policy_net(many_frame_input,3,[8,8],single = False,softmax = True)

    return {"env":env,"frame_input":frame_input,"optimizer":adam,"ECM":episodic_mem,"full_policy":many_pol_net,"policy":pol_net,"iframe":o,"many_frames":many_frame_input}

def make_cp_batch(img,n,cut = 5,kap = 2,split = .5):

    close = [[img[a],img[b]] for a in range(len(img) - cut) for b in range(a,a+cut)]
    far = [[img[a],img[b]] for a in range(len(img) - kap*cut+1) for b in range(a + kap*cut,len(img))]

    label = np.random.binomial(1,split,size = [n])

    allbatch = [close,far]
    
    batch = [allbatch[k][np.random.choice(len(allbatch[k]))] for k in label]

    ri = np.random.randint(10000)
    
    return np.array(batch),np.array(label)

def normalize(s):
    return s/np.sum(s)

def run_episode(eplen,agent_stuff,init_obs,sess,arep = 1,policy = True):
    ECM = agent_stuff["ECM"]
    o = agent_stuff["env"].reset()
    
    a_dist,femb = sess.run([agent_stuff["policy"],ECM.frame_embedding],{agent_stuff["frame_input"]:agent_stuff["iframe"]})

    obs = []
    action = []
    pact = []
    erew = []
    embed = []
    
    for k in range(eplen - np.random.randint(10)):
        obs.append(o)
        embed.append(femb)
        if policy:
            act = np.random.choice(range(3),p = a_dist)
        else:
            act = np.random.choice(range(3))

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
            break

    o = agent_stuff["env"].reset()
    return obs,action,pact,erew,embed

def KL(p1,p2,eps = .0001,axis = -1):
    return tf.reduce_sum(p1 * (tf.log(p1 + eps) - tf.log(p2 + eps)),axis = axis)
    
def main():

    agent_stuff = make_system()
    ECM = agent_stuff["ECM"]
    
    eplen = 200
    cpepc = 100
    epcepc = 500
    
    extrinsic_reward = []

    alpha = 1
    beta = 0

    action_inp = tf.placeholder(tf.float32,[None,3])
    action_prob = tf.placeholder(tf.float32,[None,3])
    reward_inp = tf.placeholder(tf.float32,[None])

    beta = 100.

#    policy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = action_inp,logits = agent_stuff["full_policy"])*reward_inp)
    policy_loss = tf.reduce_sum(tf.reduce_sum(action_inp*agent_stuff["full_policy"]/(action_prob + .001),axis = -1)*reward_inp - beta * KL(agent_stuff["full_policy"],action_prob))

    policy_train = agent_stuff["optimizer"].minimize(-policy_loss)

    init_obs = agent_stuff["iframe"]

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    ECMlen = 0
    necm = 0

    for rep in range(100):
        
        for epoch in range(cpepc if rep > 0 else 1000):
            obs,action,pact,erew,embed = run_episode(eplen,agent_stuff,init_obs,sess,policy = True)
            init_obs = obs[-1]
            cp_batch, cp_label = make_cp_batch(embed,ECM.nbatch)
            
            _,cout,cscore = sess.run([ECM.comp_train,ECM.comp_net_out,ECM.comploss],{ECM.frame1:cp_batch[:,0],ECM.frame2:cp_batch[:,1],ECM.complabel:cp_label})
            print("CP Epoch: {}\t{}".format(epoch,np.mean(cscore)))
                
        for epoch in range(epcepc):
            obs,action,pact,erew,embed = run_episode(eplen,agent_stuff,init_obs,sess,policy = True)
            init_obs = obs[-1]

            print("ECM Epoch {}".format(epoch))
            cp_out = ECM.run_ECM_on_video(obs,sess)
            
            r_int = cp_out#,alpha,beta)
            
            if len(ECM.buff) > 0:
                action_1hot = np.zeros([len(action),3])
                action_1hot[np.arange(len(action)),action] = 1.
                drew = get_discounted_rew(r_int + np.array(erew))
                for k in range(200):
                
                    full_p,policy_perf,_ = sess.run([agent_stuff["full_policy"],policy_loss,policy_train],{action_inp : action_1hot,agent_stuff["many_frames"] : np.array(obs), reward_inp : drew,action_prob: pact})
                    if k > 0:
                        if np.mean(np.abs(full_p - full_p_old)) < .00001:
                            break
                        else:
                            full_p_old = full_p
                    else:
                        full_p_old = full_p

                
                print("Performance: {}\t{}".format(policy_perf,len(ECM.buff)))
                print("Mean Reward: {}\t{}".format(np.mean(r_int + np.array(erew)),np.sum(cp_out)))
                print("Mean extrinsic reward: {}".format(np.mean(np.array(erew))))
                
                extrinsic_reward.append(np.mean(np.array(erew)))

            if epoch % 10 == 0:
                test = np.array([[a,b] for a in np.linspace(-1,0,20) for b in np.linspace(-.1,.1,20)])

                print(agent_stuff["many_frames"].shape)
                print(test.shape)

                ap = sess.run(agent_stuff["full_policy"],{agent_stuff["many_frames"]:test})

                np.savetxt("./saved_obs/aprob_{}_{}.csv".format(rep,epoch),ap)
                np.savetxt("./saved_obs/aloc_{}_{}.csv".format(rep,epoch),test)
                
            if len(ECM.buff) > 0 or true:
                print("Saved")
                np.savetxt("./saved_obs/sample_videos_{}.csv".format(epoch),np.reshape(np.array(obs),[len(obs),-1]))
                np.savetxt("./saved_obs/reward_{}.csv".format(epoch),np.reshape(drew,[len(obs),-1]))
                np.savetxt("./saved_obs/CPO_{}.csv".format(epoch),np.reshape(cp_out,[len(obs),-1]))
                                                
            ECM.clean_buffer()
    np.savetxt("./ext_rew.csv",extrinsic_reward)
            
def get_intrinsic_rew(comparator_output,a,b):
    return a * np.sum(np.array(comparator_output) - b)

def dis_sum(x,gamma):
    val = x[-1]

    for k in reversed(range(0,len(x)-1)):
        val = gamma*val + x[k]

    return val
    
def get_discounted_rew(rew,gamma = .99):

    o = []

    for k in range(len(rew)):
        o.append(dis_sum(rew[k:],gamma))

    return np.array(o)

if __name__ == "__main__":
    main()
        
