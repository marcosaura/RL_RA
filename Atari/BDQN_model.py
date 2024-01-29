from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import gym
import math
from collections import namedtuple
import time
import pickle
import logging, logging.handlers
import matplotlib.ticker as mtick

from RA_unoptimised import RingAttractor
# from RA_1 import RingAttractor

import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool, Float32, Int16

command = 'mkdir data' # Creat a direcotry to store models and scores.
os.system(command)


class Options:
    def __init__(self, skip=4, frame_stack=4):
        #Articheture
        self.batch_size = 32 # The size of the batch to learn the Q-function
        self.image_size = 84 # Resize the raw input frame to square frame of size 80 by 80 
        #Trickes
        self.replay_buffer_size = 30000 # The size of replay buffer; set it to size of your memory (.5M for 50G available memory)
        self.learning_frequency = 4 # With Freq of 1/4 step update the Q-network
        self.skip_frame = skip # Skip 4-1 raw frames between steps
        self.internal_skip_frame = skip # Skip 4-1 raw frames between skipped frames
        self.frame_len = frame_stack # Each state is formed as a concatination 4 step frames [f(t-12),f(t-8),f(t-4),f(t)]
        self.Target_update = 10000 # Update the target network each 10000 steps
        self.epsilon_min = 0.1 # Minimum level of stochasticity of policy (epsilon)-greedy
        self.annealing_end = 1000. # The number of step it take to linearly anneal the epsilon to it min value
        self.gamma = 0.99 # The discount factor
        self.replay_start_size = 10000 # Start to backpropagated through the network, learning starts
        
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
        self.target_batch_size = 10000 #target update sample batch
        self.BayesBatch = 2000 #size of batch for udpating E_W and Cov
        self.target_W_update = 10
        self.lambda_W = 0.1 #update on W = lambda W_new + (1-lambda) W
        self.sigma = 0.001 # W prior variance
        self.sigma_n = 1 # noise variacne
        self.sampled_weights_matrix_size = 10 # Number of sample distributions to compute variance 

opt = Options(skip=2)

class Replay_Buffer():
    def __init__(self, replay_buffer_size):
        self.replay_buffer_size = replay_buffer_size
        self.memory = []
        self.position = 0
    def push(self, Transition, *args):
        if len(self.memory) < self.replay_buffer_size:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.replay_buffer_size
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
class DQN_node(Node):
        
    def __init__(self, pub_namespace="/TUoM/BDQN_RA"):

        super().__init__('BDQNRA',allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)

        self.get_logger().info(str("BDQN RA NODE STARTED" ))

        self.pub_namespace = pub_namespace

        self.init_publishers()


    def init_publishers(self):
    
        self.DQN_time_publisher = self.create_publisher(Float32, self.pub_namespace + "/time",  10)
        self.DQN_episode_publisher = self.create_publisher(Float32, self.pub_namespace + "/episode",  10)
        self.DQN_frame_publisher = self.create_publisher(Float32, self.pub_namespace + "/frame_number",  10)

        self.DQN_cm_reward_publisher = self.create_publisher(Float32, self.pub_namespace + "/cummulative_reward",  10)
        self.DQN_cm_cl_reward_publisher = self.create_publisher(Float32,self.pub_namespace +  "/cummulative_clipped_reward",  10)
        self.DQN_tot_reward_publisher = self.create_publisher(Float32,self.pub_namespace +  "/total_reward",  10)
        self.DQN_tot_cl_reward_publisher = self.create_publisher(Float32, self.pub_namespace + "/total_clipped_reward",  10)

        self.DQN_action_publisher_list = []
        self.DQN_action_rew_publisher_list = []
        self.DQN_action_sigma_publisher_list = []

        self.DQN_action_rew_norm_publisher_list = []
        self.DQN_action_sigma_norm_publisher_list = []

        for action_pub in range (12):
            self.DQN_action_rew_publisher = self.create_publisher(Float32,self.pub_namespace +  "/action_reward_" + str(action_pub),  10)
            self.DQN_action_sigma_publisher = self.create_publisher(Float32, self.pub_namespace +  "/action_sigma_" + str(action_pub),  10)
            
            self.DQN_action_rew_publisher_list.append(self.DQN_action_rew_publisher)
            self.DQN_action_sigma_publisher_list.append(self.DQN_action_sigma_publisher)

        for action_pub in range (12):
            self.DQN_action_rew_publisher = self.create_publisher(Float32, self.pub_namespace + "/norm_action_reward_" + str(action_pub),  10)
            self.DQN_action_sigma_publisher = self.create_publisher(Float32,  self.pub_namespace +  "/norm_action_sigma_" + str(action_pub),  10)
            
            self.DQN_action_rew_norm_publisher_list.append(self.DQN_action_rew_publisher)
            self.DQN_action_sigma_norm_publisher_list.append(self.DQN_action_sigma_publisher)

        self.DQN_action_RA_publisher = self.create_publisher(Float32, self.pub_namespace +  "/RA_action",  10)
        self.DQN_orientation_RA_publisher = self.create_publisher(Float32,self.pub_namespace +  "/RA_orientation",  10)

        self.time_msg = Float32()
        self.episode_msg = Float32()
        self.frame_msg = Float32()
        self.cm_reward_msg = Float32()
        self.cm_cl_reward_msg = Float32()
        self.tot_reward_msg = Float32()
        self.tot_cl_reward_msg = Float32()

        self.action_msg = Float32()
        self.action_rew_msg = Float32()
        self.action_var_msg = Float32()

        self.ra_action_msg = Float32()
        self.ra_orientation_msg = Float32()


    def publish_actions(self, action, reward, variance):

        publisher_rew = self.DQN_action_rew_publisher_list[action]
        publisher_sigma = self.DQN_action_sigma_publisher_list[action]

        self.action_rew_msg.data = float(reward)
        self.action_var_msg.data = float(variance)

        publisher_rew.publish(self.action_rew_msg)
        publisher_sigma.publish(self.action_var_msg)


    def publish_actions_normalised(self, action, reward, variance):

        publisher_rew = self.DQN_action_rew_norm_publisher_list[action]
        publisher_sigma = self.DQN_action_sigma_norm_publisher_list[action]

        self.action_rew_msg.data = float(reward)
        self.action_var_msg.data = float(variance)

        publisher_rew.publish(self.action_rew_msg)
        publisher_sigma.publish(self.action_var_msg)

    
    def publish_actions_selection(self, action):


        self.ra_action_msg.data = float(action)

        self.DQN_action_RA_publisher.publish(self.ra_action_msg)
        # self.get_logger().info(str("CANT PUBLISH ACTIONS, MALFORMED DATA" ))


    def publish_results(self, epis, duration, fnum, cum_cl_rew, cum_rew, tot_cl, tot):
            self.time_msg.data = duration
            self.episode_msg.data = epis

            self.cm_reward_msg.data = cum_rew
            self.cm_cl_reward_msg.data = cum_cl_rew
            self.tot_reward_msg.data = tot
            self.tot_cl_reward_msg.data = tot_cl

            self.DQN_time_publisher.publish(self.time_msg)
            self.DQN_episode_publisher.publish(self.episode_msg)

            try:
                self.frame_msg.data = fnum
                self.DQN_frame_publisher.publish(self.frame_msg)
            except:
                self.get_logger().info(str("ROS FRAME PUBLISHER FAILLURE DUE TO LONG DATA DATA" ))

            self.DQN_cm_reward_publisher.publish(self.cm_reward_msg)
            self.DQN_cm_cl_reward_publisher.publish(self.cm_cl_reward_msg)
            self.DQN_tot_reward_publisher.publish(self.tot_reward_msg)
            self.DQN_tot_cl_reward_publisher.publish(self.tot_cl_reward_msg)


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

def BayesReg(phiphiT,phiY,alpha,batch_size, dqn_, replay_memory, bat_state, bat_state_next, E_W_target, target_dqn_, E_W, Cov_W, eye, num_action):
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
        inv = np.linalg.inv((phiphiT[i]/opt.sigma_n + 1/opt.sigma*eye).asnumpy())
        E_W[i] = nd.array(np.dot(inv,phiY[i].asnumpy())/opt.sigma_n, ctx = opt.ctx)
        Cov_W[i] = opt.sigma * inv
    return phiphiT,phiY,E_W,Cov_W 

# Thompson sampling, sample model W form the posterior.
def sample_W(E_W, E_W_M_, U, num_action):
    for i in range(num_action):
        sam = nd.normal(loc=0, scale=1, shape=(opt.lastlayer,1),ctx = opt.ctx)
        E_W_M_[i] = E_W[i] + nd.dot(U[i],sam)[:,0]
    return E_W_M_

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
        plt.imshow(next_frame)
        plt.show()
        display.clear_output(wait=True)
        time.sleep(.1)
        

def train_BDQN(env, num_action, use_RA=True, action_degrees = [0, 45, 90, 135, 180, 225, 270, 315], action_space = [], continuous = False, mario_flag=False):
    opt = Options()
    if use_RA is True:
        print("USING RING ATTRACTOR")
    else:
        print("USING BASELINE MODEL")

    if mario_flag is True:
        print("USING MARIO ENVIRONMENT")
    else:
        print("NOT USING MARIO ENVIRONMENT")

    if continuous is True:
        print("USING CONTINUOUS")
    else:
        print("USING DISCRETE")

    env_name = "generic"

    manualSeed = 1 # random.randint(1, 10000) # Set the desired seed to reproduce the results
    mx.random.seed(manualSeed)
    attrs = vars(opt)

    # set the logger
    logger = logging.getLogger()
    file_name = './data/results_BDQN.log'
    fh = logging.handlers.RotatingFileHandler(file_name)
    fh.setLevel(logging.DEBUG)#no matter what level I set here
    formatter = logging.Formatter('%(asctime)s:%(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ff =(', '.join("%s: %s" % item for item in attrs.items()))
    logging.error(str(ff))

    dqn_ = DQN_gen()
    target_dqn_ = DQN_gen()

    DQN_trainer = gluon.Trainer(dqn_.collect_params(),'RMSProp', \
                            {'learning_rate': opt.lr ,'gamma1':opt.gamma1,'gamma2': opt.gamma2,'epsilon': opt.rms_eps,'centered' : True})

    dqn_.collect_params().zero_grad()

    Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward','done'))
        
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

    E_W_M = np.empty(opt.sampled_weights_matrix_size, dtype=object) 
    for sampled_weights_matrix in range(opt.sampled_weights_matrix_size):
        E_W_M[sampled_weights_matrix] = nd.normal(loc=0, scale=.01, shape=(num_action,opt.lastlayer),ctx = opt.ctx)

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
    new_episode_write = False

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

    RA = RingAttractor()

    actions_BRL_matrix = np.empty(opt.sampled_weights_matrix_size, dtype=object) 

    actions_list = [np.zeros(3) for i in range(num_action)]
    actions_array = np.asarray(actions_list)

    rclpy.init(args=None)

    if use_RA is True: 
        pub_namespace= "BDQN_RA"
    else: 
        pub_namespace= "BDQN_baseline"

    dqn_node = DQN_node(pub_namespace=pub_namespace)

    while epis_count < opt.max_episode:
        cum_clipped_reward = 0
        print(epis_count)
        print(opt.max_episode)
        cum_reward = 0
        next_frame = env.reset()
        state = preprocess(next_frame, initial_state = True)
        t = 0.
        done = False

        truncate = False

        info = {}

        info["flag_get"] = False

        while not done or not info["flag_get"]:
            mx.nd.waitall()
            previous_state = state
            # show the frame
            # renderimage(next_frame)
            sample = random.random()
            if frame_counter > opt.replay_start_size:
                annealing_count += 1
            if frame_counter == opt.replay_start_size:
                logging.error('annealing and laerning are started ')
            
            env.render()
            data = nd.array(state.reshape([1,opt.frame_len,opt.image_size,opt.image_size]),opt.ctx)

            if use_RA ==True:

                for BRL_sample in range(len(E_W_M)):
                    actions_BRL_matrix[BRL_sample] = nd.dot(E_W_M[BRL_sample], dqn_(data)[0].T)
                miu_max = 0.000000001

                for action in range(num_action): 
                    E_V_list = []
                    for BRL_sample in range(len(E_W_M)):
                        E_V_list.append(actions_BRL_matrix[BRL_sample][action].asnumpy()[0])

                    miu = np.mean(E_V_list)
                    if abs(miu) > miu_max: miu_max = abs(miu)

                    dqn_node.publish_actions(action, miu, sigma)

                    actions_array[action][0] = miu
                    actions_array[action][1] = action_degrees[action]
                    actions_array[action][2] = 20

                actions_array[:,0] = 50*actions_array[:,0] / miu_max
                actions_array[:,0][actions_array[:,0] < -50] = -50

                # actions_array[:,2] = 50*actions_array[:,2] / sigma_max

                dqn_node.publish_actions_normalised(0,  actions_array[0][0],  actions_array[0][2])
                dqn_node.publish_actions_normalised(1,  actions_array[1][0],  actions_array[1][2])
                dqn_node.publish_actions_normalised(2,  actions_array[2][0],  actions_array[2][2])
                dqn_node.publish_actions_normalised(3,  actions_array[3][0],  actions_array[3][2])
                dqn_node.publish_actions_normalised(4,  actions_array[4][0],  actions_array[4][2])
                dqn_node.publish_actions_normalised(5,  actions_array[5][0],  actions_array[5][2])
                dqn_node.publish_actions_normalised(6,  actions_array[6][0],  actions_array[6][2])
                dqn_node.publish_actions_normalised(7,  actions_array[7][0],  actions_array[7][2])

                action_signal = RA.cue_integration(actions_array)

                action_deg = np.argmax(action_signal)*360/20

                angular_range = 360 / num_action

                action = int(round((action_deg % 360) / angular_range))
                
                if action > num_action-1: action = 0

                print(action)

                dqn_node.publish_actions_selection(action)
            else:
                data = nd.array(state.reshape([1,opt.frame_len,opt.image_size,opt.image_size]),opt.ctx)
                action = nd.dot(E_W_M[0],dqn_(data)[0].T)
                action_array = action.asnumpy()

                action = np.argmax(action_array).astype(np.uint8)

                dqn_node.publish_actions_selection(action)

            # Skip frame
            rew = 0

            for skip in range(opt.skip_frame-1):

                if continuous ==True:
                    next_frame, reward, done, info = env.step(action_space[action])
                else:
                    next_frame, reward, done, info = env.step(action)
                # renderimage(next_frame)
                cum_clipped_reward += rew_clipper(reward)
                rew += reward

                if mario_flag ==True:
                    if done or info["flag_get"]:
                        break
                else:
                    if done:
                        break
                # for internal_skip in range(opt.internal_skip_frame-1):
                #     _ , reward, done, truncate, info = env.step(action)
                #     cum_clipped_reward += rew_clipper(reward)
                #     rew += reward

            if mario_flag ==True:
                if not done and not info["flag_get"]:
                    if continuous ==True:
                        next_frame_new, reward, done, info = env.step(action_space[action])
                    else:
                        next_frame_new, reward, done, info = env.step(action)

                    # renderimage(next_frame)
                    cum_clipped_reward += rew_clipper(reward)
                    rew += reward
                    reward = rew_clipper(rew)
                    next_frame = np.maximum(next_frame_new,next_frame)
            else:
                 if not done:
                    if continuous ==True:
                        next_frame_new, reward, done, info = env.step(action_space[action])
                    else:
                        next_frame_new, reward, done, info = env.step(action)

                    # renderimage(next_frame)
                    cum_clipped_reward += rew_clipper(reward)
                    rew += reward
                    reward = rew_clipper(rew)
                    next_frame = np.maximum(next_frame_new,next_frame)
                
            cum_reward += rew
            
            # Reward clipping
            
            state = preprocess(next_frame, state)
            replay_memory.push(Transition,(previous_state*255.).astype('uint8')\
                            ,action,(state*255.).astype('uint8'),reward,done)
            # Thompson Sampling

            if frame_counter % opt.f_sampling:
                for sampled_weights_matrix in range(opt.sampled_weights_matrix_size):
                    E_W_M[sampled_weights_matrix] = sample_W(E_W,E_W_M[sampled_weights_matrix],Cov_W_decom, num_action)
                
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
                        argmax_Q = nd.argmax(nd.dot(dqn_(batch_state_next),((E_W_M[0]+E_W_M[1]+E_W_M[2]+E_W_M[3]+E_W_M[4]+E_W_M[5]+E_W_M[6]+E_W_M[7]+E_W_M[8]+E_W_M[9])/10).T),axis = 1).astype('int32')
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
                        phiphiT,phiY,E_W,Cov_W = BayesReg(phiphiT,phiY,opt.alpha,opt.batch_size, dqn_, replay_memory, bat_state, bat_state_next, E_W_target, target_dqn_, E_W, Cov_W, eye, num_action)
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
            
            if mario_flag ==True:
                if done or info["flag_get"]:
                    break
            else:
                if done:
                    break

        epis_count += 1
        print('BDQN:env:%s,epis[%d],durat[%d],fnum=%d, cum_cl_rew = %d, cum_rew = %d,tot_cl = %d , tot = %d'\
                %(env_name, epis_count,t+1,frame_counter,cum_clipped_reward,cum_reward,moving_average_clipped,moving_average))

        dqn_node.publish_results(epis_count,t+1,frame_counter,cum_clipped_reward,cum_reward,moving_average_clipped,moving_average)

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


    plt.ticklabel_format(axis='both', style='sci', scilimits=(-2,2),fontsize=1, family = 'serif')
    plt.legend(fontsize=1)
    print('Running after %d number of episodes' %epis_count)
    plt.xlabel("Number of steps",fontsize=1, family = 'serif')
    plt.ylabel("Average Reward per episode",fontsize=1, family = 'serif')
    plt.title("%s" %(env_name),fontsize=1, family = 'serif')
    plt.show()

if __name__ == "__main__": 

    env_name = 'AsterixNoFrameskip-v4' # Set the desired environment#

    if gym.__version__ < '0.26':
        env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    else:
        env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", apply_api_compatibility=True)

    env = JoypadSpace(env, [["right"], ["right", "A"], ["A"], ["left","A"], ["left"], ["left", 'down'], ['down'], ['down', "right"]])
    num_action = env.action_space.n # Extract the number of available action from the environment setting

    train_BDQN(env, num_action, use_RA=False)
