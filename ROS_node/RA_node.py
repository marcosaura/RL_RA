
import random
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from tqdm import tqdm
import pickle 
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
import numpy as np

import time

from RA_unoptimised import RingAttractor

import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool, Float32, Int16
from std_msgs.msg import Float32MultiArray
import random


class RA_node(Node):
        
    def __init__(self, namespace):

        super().__init__('DDQNRA',allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)

        self.get_logger().info(str("DDQN RA NODE STARTED" ))

        self.namespace = namespace

        self.init_publishers()


    def init_publishers(self):
    
        self.DQN_time_publisher = self.create_publisher(Float32, self.namespace + "/time",  10)
        self.DQN_episode_publisher = self.create_publisher(Float32, self.namespace +  "/episode",  10)
        self.DQN_frame_publisher = self.create_publisher(Float32, self.namespace +  "/frame_number",  10)

        self.DQN_cm_reward_publisher = self.create_publisher(Float32, self.namespace +  "/cummulative_reward",  10)
        self.DQN_cm_cl_reward_publisher = self.create_publisher(Float32, self.namespace +  "/cummulative_clipped_reward",  10)
        self.DQN_tot_reward_publisher = self.create_publisher(Float32,  self.namespace + "/total_reward",  10)
        self.DQN_tot_cl_reward_publisher = self.create_publisher(Float32, self.namespace +  "/total_clipped_reward",  10)

        self.DQN_action_publisher_list = []
        self.DQN_action_rew_publisher_list = []
        self.DQN_action_sigma_publisher_list = []

        self.DQN_action_rew_norm_publisher_list = []
        self.DQN_action_sigma_norm_publisher_list = []

        for action_pub in range (8):
            self.DQN_action_rew_publisher = self.create_publisher(Float32, self.namespace +  "/action_reward_" + str(action_pub),  10)
            self.DQN_action_sigma_publisher = self.create_publisher(Float32, self.namespace +  "/action_sigma_" + str(action_pub),  10)
            
            self.DQN_action_rew_publisher_list.append(self.DQN_action_rew_publisher)
            self.DQN_action_sigma_publisher_list.append(self.DQN_action_sigma_publisher)

        for action_pub in range (8):
            self.DQN_action_rew_publisher = self.create_publisher(Float32, self.namespace +  "/norm_action_reward_" + str(action_pub),  10)
            self.DQN_action_sigma_publisher = self.create_publisher(Float32, self.namespace +  "/norm_action_sigma_" + str(action_pub),  10)
            
            self.DQN_action_rew_norm_publisher_list.append(self.DQN_action_rew_publisher)
            self.DQN_action_sigma_norm_publisher_list.append(self.DQN_action_sigma_publisher)

        self.DQN_action_RA_publisher = self.create_publisher(Float32, self.namespace +  "/RA_action",  10)
        self.DQN_orientation_RA_publisher = self.create_publisher(Float32,  self.namespace +  "/RA_orientation",  10)

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

    
    def publish_actions_selection(self, action, orientation):

        try:
            self.ra_action_msg.data = float(action)
            self.ra_orientation_msg.data = float(orientation)

            self.DQN_action_RA_publisher.publish(self.ra_action_msg)
            self.DQN_orientation_RA_publisher.publish(self.ra_orientation_msg)
        except:
            self.get_logger().info(str("CANT PUBLISH ACTIONS, MALFORMED DATA" ))


    def publish_results(self, epis, fnum, tot):
            self.episode_msg.data = epis

            self.tot_reward_msg.data = tot

            self.DQN_episode_publisher.publish(self.episode_msg)

            self.frame_msg.data = fnum
            self.DQN_frame_publisher.publish(self.frame_msg)

            self.DQN_tot_reward_publisher.publish(self.tot_reward_msg)


    def act(self, actions_values_array, ddqn_action = -1, combine_with_nn_policy = False):
                """Epsilon-greedy action"""
    
                if ddqn_action == -1:
                    ddqn_action = np.argmax(actions_values_array)

                miu_max= np.max(abs(actions_values_array))
    
                for action_n in range(len(actions_values_array)): 
                    miu = actions_values_array[action_n] #+ actions_values_array[action_n]*random.uniform(0, 1)/20

                    self.actions_array[action_n][0] = miu
                    self.actions_array[action_n][1] = 90 - action_n*45
                    self.actions_array[action_n][2] = 20
                    
                    self.dqn_node.publish_actions(action_n, self.actions_array[action_n][0], self.actions_array[action_n][2])

                self.actions_array[:,0] = 30*((self.actions_array[:,0] )/(np.max(self.actions_array[:,0])))
                
                self.dqn_node.publish_actions_normalised(0,  self.actions_array[0][0],  self.actions_array[0][2])
                self.dqn_node.publish_actions_normalised(1,  self.actions_array[1][0],  self.actions_array[1][2])
                self.dqn_node.publish_actions_normalised(2,  self.actions_array[2][0],  self.actions_array[2][2])
                self.dqn_node.publish_actions_normalised(3,  self.actions_array[3][0],  self.actions_array[3][2])
                self.dqn_node.publish_actions_normalised(4,  self.actions_array[4][0],  self.actions_array[4][2])
                self.dqn_node.publish_actions_normalised(5,  self.actions_array[5][0],  self.actions_array[5][2])
                self.dqn_node.publish_actions_normalised(6,  self.actions_array[6][0],  self.actions_array[6][2])
                self.dqn_node.publish_actions_normalised(7,  self.actions_array[7][0],  self.actions_array[7][2])
                
                action_signal = self.RA.cue_integration(self.actions_array)

                action_signal = np.argmax(action_signal)*360/20
                action = int(round(10 -action_signal/45))

                if action > 7: action = action - 8

                self.dqn_node.publish_actions_selection(int(action), action_signal)

                if combine_with_nn_policy is True:  action = random.choice([ddqn_action, action])

                return action