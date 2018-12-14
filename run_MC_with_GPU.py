import numpy as np
import tensorflow as tf
import gym
import numpy as np
import episodic_curiosity.episodic_curiosity_module as ECM_build
import agent.policy_network as policy
import episodic_curiosity.comparator_network as comp
import episodic_curiosity.embeding as emb
import os

os.environ["CUDA_VISIBLE_DEVICES"]="2"

np.random.seed(1)
tf.set_random_seed(0)

def make_system():

    env = gym.make("MountainCar-v0")

    o = env.reset()

    memmax = 200

    embsize = 8
    
    frame_input = tf.placeholder(tf.float32,o.shape)
    many_frame_input = tf.placeholder(tf.float32,[None]+ list(o.shape))
    many_frame_input_2 = tf.placeholder(tf.float32,[None]+ list(o.shape))
    adam = tf.train.AdamOptimizer()

    pol_net = policy.make_dense_policy_net(frame_input,3,[8,8],single = True,softmax = False)
    many_pol_net = policy.make_dense_policy_net(many_frame_input,3,[8,8],single = False,softmax = False)

    emb_net = emb.dense_embedding(frame_input,{"emb_size":embsize},single = True)
    many_emb_net =emb.dense_embedding(many_frame_input,{"emb_size":embsize},single = False)
    many_emb_net_2 =emb.dense_embedding(many_frame_input_2,{"emb_size":embsize},single = False)

    comp_net = comp.get_comparator(many_emb_net,many_emb_net_2)
    
    episodic_mem = ECM_build.novelty_bonus_module(emb_net,many_emb_net,comp.get_comparator,memmax)

    return {"env":env,"frame_input":frame_input,"many_frames":many_frame_input,"many_frames_2":many_frame_input_2,"novelty_module":episodic_mem,"optimizer":adam,"full_policy":many_pol_net,"policy":pol_net,"iframe":o,"embedding":emb_net,"many_embedding":many_emb_net,"comp_net_out":comp_net,"norm_probs":tf.nn.softmax(pol_net)}

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

def run_episode(eplen,agent_stuff,sess,arep = 1,policy = True):
    ECM = agent_stuff["novelty_module"]
    emb = agent_stuff["embedding"]
    
    o = agent_stuff["env"].reset()
    
    a_dist,femb = sess.run([agent_stuff["norm_probs"],emb],{agent_stuff["frame_input"]:o})

    obs = []
    action = []
    pact = []
    erew = []
    embed = []
    
    for k in range(eplen):
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

        a_dist,femb = sess.run([agent_stuff["norm_probs"],emb],{agent_stuff["frame_input"]:o})

        if done:
            break

    return obs,action,pact,erew,embed

def KL(p1,p2,axis = -1):
    '''
    Description: calculates the KL divergence between two unnormalized log-probabilities.

    args:
    p1 - first distribution 
    p2 - second distribution
    axis - axis over which to calculate KL

    returns:
    float32 - the KL divergence

    Note: This function assumes UNNORMALIZED LOG-PROBABILITIES. This function computes a numerically stable, normalized KL divergence via the identity: $\sum_{i} \exp(p_{i}) = \exp(b)\sum_{i} \exp(p_{i} - b)$. By taking b = max(p) we can combine this identity along with the unnormalized log probabilities `p` to compute provide numerical stability in spite of large/small log-probabilities. In particular, the probability distribution encoded by unnormalized log-probabilities is indepenent of an overall shift in all the values, so we remove the largest value to stabilze
    '''

    p1stable = normalized_log_proba(p1)
    p2stable = normalized_log_proba(p2)
    
    return tf.reduce_sum(tf.exp(p1stable) * (p1stable - p2stable),axis = axis)

def normalized_log_proba(p,axis = -1):
    '''
    Description: Calculate the normalized log-probability from an unnormalized log-probability in a stable manner.
    
    args:
    p - the unnormalized log-probability
    axis - the axis along which to normalize

    return
    float32 - normalized log probability

    '''
    pmax = tf.reduce_max(p,axis = axis,keepdims = True)
    pnorm = tf.log(tf.reduce_sum(tf.exp(p - pmax),axis = axis, keepdims = True))
    return p - pmax - pnorm

def log_loss(lp,lab,eps = .01):
    return -(tf.log(eps + (1. - eps)*lp)*lab + tf.log(1. - (1. - eps)*lp)*(1 - lab))

def main():

    agent_stuff = make_system()
    ECM = agent_stuff["novelty_module"]
    
    eplen = 200
    cpepc = 100
    epcepc = 500
    
    extrinsic_reward = []

    alpha = 1
    beta = 0

    action_inp = tf.placeholder(tf.float32,[None,3],name = "action")
    action_prob = tf.placeholder(tf.float32,[None,3],name = "action_prob")
    reward_inp = tf.placeholder(tf.float32,[None],name = "reward")
    comp_label = tf.placeholder(tf.float32,[None,1],name = "comparator_label")

    beta = 100.
#    policy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = action_inp,logits = agent_stuff["full_policy"])*reward_inp)
    policy_loss = tf.reduce_sum(tf.reduce_sum(action_inp*tf.exp(agent_stuff["full_policy"] - action_prob),axis = -1)*reward_inp - beta * KL(agent_stuff["full_policy"],action_prob))

    policy_train = agent_stuff["optimizer"].minimize(-policy_loss)

    comp_loss = log_loss(agent_stuff["comp_net_out"],comp_label)
    comp_train = agent_stuff["optimizer"].minimize(comp_loss)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    ECMlen = 0
    necm = 0

    for rep in range(10):
        
        for epoch in range(cpepc if rep > 0 else 1):
            obs,action,pact,erew,embed = run_episode(eplen,agent_stuff,sess,policy = True)
            cp_batch, cp_label = make_cp_batch(obs,32)
            
            _,cpo,cscore = sess.run([comp_train,agent_stuff["comp_net_out"],comp_loss],{agent_stuff["many_frames"]:cp_batch[:,0],agent_stuff["many_frames_2"]:cp_batch[:,1],comp_label:np.expand_dims(cp_label,-1)})
            
            print("CP Epoch: {}\t{}\t{}".format(epoch,np.mean(cscore),np.mean(cpo)))
                
        for epoch in range(epcepc):
            print("ECM Epoch {}".format(epoch))

            obs,action,pact,erew,embed = run_episode(eplen,agent_stuff,sess,policy = True)

            r_int = sess.run(agent_stuff["novelty_module"].score_episode_op,{agent_stuff["many_frames"]:obs})
            print(r_int)
                     
            if True:
                action_1hot = np.zeros([len(action),3])
                action_1hot[np.arange(len(action)),action] = 1.
                drew = get_discounted_rew(r_int + np.array(erew))
                
                for k in range(200):
                    full_p,policy_perf,_ = sess.run([agent_stuff["full_policy"],policy_loss,policy_train],{action_inp : action_1hot,agent_stuff["many_frames"] : np.array(obs), action_prob: pact,reward_inp:drew})
                    
                    if k > 0:
                        if np.mean(np.abs(full_p - full_p_old)) < .00001:
                            break
                        else:
                            full_p_old = full_p
                    else:
                        full_p_old = full_p

                
                print("Performance: {}\t{}".format(policy_perf,sess.run(ECM.memory.nmem)))
                print("Mean Reward: {}".format(np.mean(drew)))
                print("Mean extrinsic reward: {}".format(np.mean(np.array(erew))))
                
                extrinsic_reward.append(np.mean(np.array(erew)))

            if epoch % 10 == 0:
                test = np.array([[a,b] for a in np.linspace(-1,0,20) for b in np.linspace(-.1,.1,20)])

                ap = sess.run(agent_stuff["full_policy"],{agent_stuff["many_frames"]:test})

                np.savetxt("./saved_obs/aprob_{}_{}.csv".format(rep,epoch),ap)
                np.savetxt("./saved_obs/aloc_{}_{}.csv".format(rep,epoch),test)
                
        
            np.savetxt("./saved_obs/sample_videos_{}.csv".format(epoch),np.reshape(np.array(obs),[len(obs),-1]))
            np.savetxt("./saved_obs/reward_{}.csv".format(epoch),np.reshape(drew,[len(obs),-1]))
                                                
            ECM.memory.clear_memory(sess)
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
        
