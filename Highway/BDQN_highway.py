from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon

import highway_env
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import gymnasium as gym
import math
from collections import namedtuple
import time
import pickle
import logging, logging.handlers
import matplotlib.ticker as mtick


command = 'mkdir data' # Creat a direcotry to store models and scores.
os.system(command)
class Options:
    def __init__(self):
        #Articheture
        self.batch_size = 32 # The size of the batch to learn the Q-function
        self.image_size = 84 # Resize the raw input frame to square frame of size 80 by 80 
        #Trickes
        self.replay_buffer_size = 30000 # The size of replay buffer; set it to size of your memory (.5M for 50G available memory)
        self.learning_frequency = 4 # With Freq of 1/4 step update the Q-network
        self.skip_frame = 0 # Skip 4-1 raw frames between steps
        self.internal_skip_frame = 0 # Skip 4-1 raw frames between skipped frames
        self.frame_len = 4 # Each state is formed as a concatination 4 step frames [f(t-12),f(t-8),f(t-4),f(t)]
        self.Target_update = 1000 # Update the target network each 10000 steps
        self.epsilon_min = 0.1 # Minimum level of stochasticity of policy (epsilon)-greedy
        self.annealing_end = 100. # The number of step it take to linearly anneal the epsilon to it min value
        self.gamma = 0.99 # The discount factor
        self.replay_start_size = 200 # Start to backpropagated through the network, learning starts
        
        #otimization
        self.max_episode =   20000 #max number of episodes#
        self.lr = 0.0025 # RMSprop learning rate
        self.gamma1 = 0.95 # RMSprop gamma1
        self.gamma2 = 0.95 # RMSprop gamma2
        self.rms_eps = 0.01 # RMSprop epsilon bias
        self.ctx = mx.gpu() # Enables gpu if available, if not, set it to mx.cpu()
        self.lastlayer = 512 # Dimensionality of feature space
        self.f_sampling = 500 # frequency sampling E_W_ (Thompson Sampling)
        self.alpha = .01 # forgetting factor 1->forget
        self.alpha_target = 1 # forgetting factor 1->forget
        self.f_bayes_update = 500 # frequency update E_W and Cov
        self.target_batch_size = 1000 #target update sample batch
        self.BayesBatch = 1000 #size of batch for udpating E_W and Cov
        self.target_W_update = 10
        self.lambda_W = 0.1 #update on W = lambda W_new + (1-lambda) W
        self.sigma = 0.001 # W prior variance
        self.sigma_n = 1 # noise variacne
opt = Options()

env_name = "highway"
env = gym.make('highway-v0', render_mode='rgb_array')
config = {
       "observation": {
           "type": "GrayscaleObservation",
           "observation_shape": (84, 84),
           "stack_size": 4,
           "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
           "scaling": 1.75,
       },
       "policy_frequency": 2,
        "action": {
        "type": "ContinuousAction"
    }
   }

env.configure(config)
env.reset()
print(env.action_space.sample())
num_action = 8 # Extract the number of available action from the environment setting

manualSeed = 1 # random.randint(1, 10000) # Set the desired seed to reproduce the results
mx.random.seed(manualSeed)
attrs = vars(opt)

# set the logger
logger = logging.getLogger()
file_name = './data/results_BDQN_%s_lr_%f.log' %(env_name,opt.lr)
fh = logging.handlers.RotatingFileHandler(file_name)
fh.setLevel(logging.DEBUG)#no matter what level I set here
formatter = logging.Formatter('%(asctime)s:%(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

ff =(', '.join("%s: %s" % item for item in attrs.items()))
logging.error(str(ff))
def DQN_gen():
    DQN = gluon.nn.Sequential()
    with DQN.name_scope():
        #first layer
        DQN.add(gluon.nn.Conv2D(channels=32, kernel_size=8,strides = 4,padding = 0))
        DQN.add(gluon.nn.BatchNorm(axis = 1, momentum = 0.1,center=True))
        DQN.add(gluon.nn.Activation('relu'))
        #second layer
        DQN.add(gluon.nn.Conv2D(channels=64, kernel_size=4,strides = 2))
        DQN.add(gluon.nn.BatchNorm(axis = 1, momentum = 0.1,center=True))
        DQN.add(gluon.nn.Activation('relu'))
        #tird layer
        DQN.add(gluon.nn.Conv2D(channels=64, kernel_size=3,strides = 1))
        DQN.add(gluon.nn.BatchNorm(axis = 1, momentum = 0.1,center=True))
        DQN.add(gluon.nn.Activation('relu'))
        DQN.add(gluon.nn.Flatten())
        #fourth layer
        #fifth layer
        DQN.add(gluon.nn.Dense(opt.lastlayer,activation ='relu'))
    DQN.collect_params().initialize(mx.init.Normal(0.02), ctx=opt.ctx)
    return DQN

dqn_ = DQN_gen()
target_dqn_ = DQN_gen()

DQN_trainer = gluon.Trainer(dqn_.collect_params(),'RMSProp', \
                          {'learning_rate': opt.lr ,'gamma1':opt.gamma1,'gamma2': opt.gamma2,'epsilon': opt.rms_eps,'centered' : True})
dqn_.collect_params().zero_grad()
Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward','done'))
class Replay_Buffer():
    def __init__(self, replay_buffer_size):
        self.replay_buffer_size = replay_buffer_size
        self.memory = []
        self.position = 0
    def push(self, *args):
        if len(self.memory) < self.replay_buffer_size:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.replay_buffer_size
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
bat_state = nd.empty((1,opt.frame_len,opt.image_size,opt.image_size), opt.ctx)
bat_state_next = nd.empty((1,opt.frame_len,opt.image_size,opt.image_size), opt.ctx)
bat_reward = nd.empty((1), opt.ctx)
bat_action = nd.empty((1), opt.ctx)
bat_done = nd.empty((1), opt.ctx)

eye = nd.zeros((opt.lastlayer,opt.lastlayer), opt.ctx)
for i in range(opt.lastlayer):
    eye[i,i] = 1

E_W = nd.normal(loc=0, scale=.01, shape=(num_action,opt.lastlayer),ctx = opt.ctx)
E_W_target = nd.normal(loc=0, scale=.01, shape=(num_action,opt.lastlayer),ctx = opt.ctx)
E_W_ = nd.normal(loc=0, scale=.01, shape=(num_action,opt.lastlayer),ctx = opt.ctx)
Cov_W = nd.normal(loc=0, scale= 1, shape=(num_action,opt.lastlayer,opt.lastlayer),ctx = opt.ctx)+eye
Cov_W_decom = Cov_W
for i in range(num_action):
    Cov_W[i] = eye
    Cov_W_decom[i] = nd.array(np.linalg.cholesky(((Cov_W[i]+nd.transpose(Cov_W[i]))/2.).asnumpy()),ctx = opt.ctx)
Cov_W_target = Cov_W
phiphiT = nd.zeros((num_action,opt.lastlayer,opt.lastlayer), opt.ctx)
phiY = nd.zeros((num_action,opt.lastlayer), opt.ctx)
sigma = opt.sigma
sigma_n = opt.sigma_n

def BayesReg(phiphiT,phiY,alpha,batch_size):
    phiphiT *= (1-alpha) #Forgetting parameter alpha suggest how much of the moment from the past can be used, we set alpha to 1 which means do not use the past moment
    phiY *= (1-alpha)
    for j in range(batch_size):
        transitions = replay_memory.sample(1) # sample a minibatch of size one from replay buffer
        bat_state[0] = transitions[0].state.as_in_context(opt.ctx).astype('float32')/255.
        bat_state_next[0] = transitions[0].next_state.as_in_context(opt.ctx).astype('float32')/255.
        bat_reward = transitions[0].reward 
        bat_action = transitions[0].action 
        bat_done = transitions[0].done 
        phiphiT[int(bat_action)] += nd.dot(dqn_(bat_state).T,dqn_(bat_state))
        phiY[int(bat_action)] += (dqn_(bat_state)[0].T*(bat_reward +(1.-bat_done) * opt.gamma * nd.max(nd.dot(E_W_target,target_dqn_(bat_state_next)[0].T))))
    for i in range(num_action):
        inv = np.linalg.inv((phiphiT[i]/sigma_n + 1/sigma*eye).asnumpy())
        E_W[i] = nd.array(np.dot(inv,phiY[i].asnumpy())/sigma_n, ctx = opt.ctx)
        Cov_W[i] = sigma * inv
    return phiphiT,phiY,E_W,Cov_W 

# Thompson sampling, sample model W form the posterior.
def sample_W(E_W,U):
    for i in range(num_action):
        sam = nd.normal(loc=0, scale=1, shape=(opt.lastlayer,1),ctx = opt.ctx)
        E_W_[i] = E_W[i] + nd.dot(U[i],sam)[:,0]
    return E_W_

def preprocess(raw_frame, currentState = None, initial_state = False):
    raw_frame = nd.array(raw_frame,mx.cpu())
    raw_frame = nd.reshape(nd.mean(raw_frame, axis = 2),shape = (raw_frame.shape[0],raw_frame.shape[1],1))
    raw_frame = mx.image.imresize(raw_frame,  opt.image_size, opt.image_size)
    raw_frame = nd.transpose(raw_frame, (2,0,1))
    raw_frame = raw_frame.astype('float32')/255.
    if initial_state == True:
        state = raw_frame
        for _ in range(opt.frame_len-1):
            state = nd.concat(state , raw_frame, dim = 0)
    else:
        state = mx.nd.concat(currentState[1:,:,:], raw_frame, dim = 0)
    return state

def rew_clipper(rew):
    if rew>0.:
        return 1.
    elif rew<0.:
        return -1.
    else:
        return 0

def renderimage(next_frame):
    if render_image:
        plt.imshow(next_frame);
        plt.show()
        display.clear_output(wait=True)
        time.sleep(.1)
        
l2loss = gluon.loss.L2Loss(batch_axis=0)
frame_counter = 0. # Counts the number of steps so far
annealing_count = 0. # Counts the number of annealing steps
epis_count = 0. # Counts the number episodes so far
replay_memory = Replay_Buffer(opt.replay_buffer_size) # Initialize the replay buffer
tot_clipped_reward = []
tot_reward = []
frame_count_record = []
moving_average_clipped = 0.
moving_average = 0.
flag = 0
c_t = 0
render_image = False # Whether to render Frames and show the game
batch_state = nd.empty((opt.batch_size,opt.frame_len,opt.image_size,opt.image_size), opt.ctx)
batch_state_next = nd.empty((opt.batch_size,opt.frame_len,opt.image_size,opt.image_size), opt.ctx)
batch_reward = nd.empty((opt.batch_size),opt.ctx)
batch_action = nd.empty((opt.batch_size),opt.ctx)
batch_done = nd.empty((opt.batch_size),opt.ctx)

def map_action(action):

    if action == 0:
        return [0, 1]
    
    if action == 1:
        return [0.5, 0.5]
    
    if action == 2:
        return [1, 0]
    
    if action == 3:
        return [0.5, -0.5]
    
    if action == 4:
        return [0, -1]
    
    if action == 5:
        return [-0.5, -0.5]
    
    if action == 6:
        return [-1, 0]
    
    if action == 7:
        return [-1, 0.5]
        

while epis_count < opt.max_episode:
    cum_clipped_reward = 0
    print(epis_count)
    print(opt.max_episode)
    cum_reward = 0
    next_frame = env.reset()[0]
    # state, reward, done, info = env.step(action=0)
    state = preprocess(next_frame, initial_state = True)
    t = 0.
    done = False


    while not done:
        mx.nd.waitall()
        previous_state = state
        # show the frame
        renderimage(next_frame)
        env.render()
        sample = random.random()
        # print(frame_counter)
        if frame_counter > opt.replay_start_size:
            annealing_count += 1
        if frame_counter == opt.replay_start_size:
            logging.error('annealing and laerning are started ')

        data = nd.array(state.reshape([1,opt.frame_len,opt.image_size,opt.image_size]),opt.ctx)
        a = nd.dot(E_W_,dqn_(data)[0].T)
        action = np.argmax(a.asnumpy()).astype(np.uint8)
        
        # Skip frame
        rew = 0
        for skip in range(opt.skip_frame-1):
            next_frame, reward, done, truncate, info = env.step(map_action(action))
            if truncate==True: done=True
            if info["on_road_reward"]==0.0: done=True
            renderimage(next_frame)
            cum_clipped_reward += rew_clipper(reward)
            rew += reward
            # for internal_skip in range(opt.internal_skip_frame-1):
            #     _ , reward, done, truncate, _ = env.step(map_action(action))
            #     cum_clipped_reward += rew_clipper(reward)
            #     rew += reward
        next_frame_new, reward,  done, truncate, info = env.step(map_action(action))
        print(info)
        if info["rewards"]["on_road_reward"]==0.0: done=True
        if info["speed"]<=0.0: done=True
        if truncate==True: done=True

        renderimage(next_frame)
        cum_clipped_reward += rew_clipper(reward)
        rew += reward
        cum_reward += rew
        
        # Reward clipping
        reward = rew_clipper(rew)
        next_frame = np.maximum(next_frame_new,next_frame)
        state = preprocess(next_frame, state)
        replay_memory.push((previous_state*255.).astype('uint8')\
                           ,action,(state*255.).astype('uint8'),reward,done)
        # Thompson Sampling

        if frame_counter % opt.f_sampling:
            E_W_ = sample_W(E_W,Cov_W_decom)
        
        # Train
        if frame_counter > opt.replay_start_size: 
            if frame_counter % opt.learning_frequency == 0:
                batch = replay_memory.sample(opt.batch_size)
                #update network
                for j in range(opt.batch_size):
                    batch_state[j] = batch[j].state.as_in_context(opt.ctx).astype('float32')/255.
                    batch_state_next[j] = batch[j].next_state.as_in_context(opt.ctx).astype('float32')/255.
                    batch_reward[j] = batch[j].reward
                    batch_action[j] = batch[j].action
                    batch_done[j] = batch[j].done
                with autograd.record():
                    argmax_Q = nd.argmax(nd.dot(dqn_(batch_state_next),E_W_.T),axis = 1).astype('int32')
                    Q_sp_ = nd.dot(target_dqn_(batch_state_next),E_W_target.T)
                    Q_sp = nd.pick(Q_sp_,argmax_Q,1) * (1 - batch_done)
                    Q_s_array = nd.dot(dqn_(batch_state),E_W.T)
                    if (Q_s_array[0,0] != Q_s_array[0,0]).asscalar():
                        flag = 1
                        print('break')
                        break
                    Q_s = nd.pick(Q_s_array,batch_action,1)
                    loss = nd.mean(l2loss(Q_s ,  (batch_reward + opt.gamma *Q_sp)))
                loss.backward()
                DQN_trainer.step(opt.batch_size)
        t += 1
        frame_counter += 1
        # Save the model, update Target model and update posterior
        if frame_counter > opt.replay_start_size:
            if frame_counter % opt.Target_update == 0 :
                check_point = frame_counter / (opt.Target_update *100)
                fdqn = './data/target_%s_%d' % (env_name,int(check_point))
                dqn_.save_params(fdqn)
                target_dqn_.load_params(fdqn, opt.ctx)
                c_t += 1
                if c_t == opt.target_W_update:
                    phiphiT,phiY,E_W,Cov_W = BayesReg(phiphiT,phiY,opt.alpha_target,opt.target_batch_size)
                    E_W_target = E_W
                    Cov_W_target = Cov_W
                    fnam = './data/clippted_rew_BDQN_%s_tarUpd_%d_lr_%f' %(env_name,opt.target_W_update,opt.lr)
                    np.save(fnam,tot_clipped_reward)
                    fnam = './data/tot_rew_BDQN_%s_tarUpd_%d_lr_%f' %(env_name,opt.target_W_update,opt.lr)
                    np.save(fnam,tot_reward)
                    fnam = './data/frame_count_BDQN_%s_tarUpd_%d_lr_%f' %(env_name,opt.target_W_update,opt.lr)
                    np.save(fnam,frame_count_record)
                    fnam = './data/E_W_target_BDQN_%s_tarUpd_%d_lr_%f_%d' %(env_name,opt.target_W_update,opt.lr,int(check_point))
                    np.save(fnam,E_W_target.asnumpy())
                    fnam = './data/Cov_W_target_BDQN_%s_tarUpd_%d_lr_%f_%d' %(env_name,opt.target_W_update,opt.lr,int(check_point))
                    np.save(fnam,Cov_W_target.asnumpy())
                    
                    c_t = 0
                    for ii in range(num_action):
                        Cov_W_decom[ii] = nd.array(np.linalg.cholesky(((Cov_W[ii]+nd.transpose(Cov_W[ii]))/2.).asnumpy()),ctx = opt.ctx)
                if len(replay_memory.memory) < 100000:
                    opt.target_batch_size = len(replay_memory.memory)
                else:
                    opt.target_batch_size = 100000
        if done:
            print("FR ", frame_counter)
            print("REW ", moving_average)
            print("CREW ", cum_reward)
            if epis_count % 2. == 0. :
                logging.error('BDQN:env:%s,epis[%d],durat[%d],fnum=%d, cum_cl_rew = %d, cum_rew = %d,tot_cl = %d , tot = %d'\
                  %(env_name, epis_count,t+1,frame_counter,cum_clipped_reward,cum_reward,moving_average_clipped,moving_average))
    epis_count += 1

    tot_clipped_reward = np.append(tot_clipped_reward, cum_clipped_reward)
    tot_reward = np.append(tot_reward, cum_reward)
    frame_count_record = np.append(frame_count_record,frame_counter)
    if epis_count > 100.:
        moving_average_clipped = np.mean(tot_clipped_reward[int(epis_count)-1-100:int(epis_count)-1])
        moving_average = np.mean(tot_reward[int(epis_count)-1-100:int(epis_count)-1])
    
    if flag:
        print('break')
        break

    tot_c = tot_clipped_reward
tot = tot_reward
fram = frame_count_record
epis_count = len(fram)


bandwidth = 1 # Moving average bandwidth
total_clipped = np.zeros(int(epis_count)-bandwidth)
total_rew = np.zeros(int(epis_count)-bandwidth)
f_num = fram[0:epis_count-bandwidth]


for i in range(int(epis_count)-bandwidth):
    total_clipped[i] = np.sum(tot_c[i:i+bandwidth])/bandwidth
    total_rew[i] = np.sum(tot[i:i+bandwidth])/bandwidth
        
    
t = np.arange(int(epis_count)-bandwidth)
belplt = plt.plot(f_num,total_rew[0:int(epis_count)-bandwidth],"b", label = "BDQN")


plt.ticklabel_format(axis='both', style='sci', scilimits=(-2,2),fontsize=fonts, family = 'serif')
plt.legend(fontsize=fonts)
print('Running after %d number of episodes' %epis_count)
plt.xlabel("Number of steps",fontsize=fonts, family = 'serif')
plt.ylabel("Average Reward per episode",fontsize=fonts, family = 'serif')
plt.title("%s" %(env_name),fontsize=fonts, family = 'serif')
plt.show()
