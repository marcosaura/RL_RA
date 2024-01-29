import gym

import gym_super_mario_bros

from nes_py.wrappers import JoypadSpace


from BDQN_model import train_BDQN


def Mario():

    env_name = 'AsterixNoFrameskip-v4' # Set the desired environment#

    if gym.__version__ < '0.26':
        env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    else:
        env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", apply_api_compatibility=True)

    env = JoypadSpace(env, [["right"], ["right", "A"], ["A"], ["left","A"], ["left"], ["left", 'down'], ['down'], ['down', "right"]])
    num_action = env.action_space.n # Extract the number of available action from the environment setting

    train_BDQN(env, num_action, use_RA=False, mario_flag=True)


def Car():
    env = gym.make('CarRacing-v0')
    # env = JoypadSpace(env,  [["right"], ["right", "A"], ["B"], ["left","A"], ["left"], ["left", 'down'], ['down'], ['down', "right"]])
    action_space    = [
            (0, 1, 0), (0.00, 0.3, 0), (0.2, 0.2, 0), #           Action Space Structure
            (1, 0.2, 0), (0.5, 0.0,   0), (0.33, 0,   0.2), #        (Steering Wheel, Gas, Break)
            (0.0, 0.0,   0.6), (-0.33, 0,   0.2), (-0.5, 0.0,   0),  # Range        -1~1       0~1   0~1
            (-1.0, 0.05,   0), (-0.2, 0.2,   0.0), (0.00, 0.3,   0.0)
        ]
    
    degrees = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    num_action = len(action_space)

    train_BDQN(env, num_action, use_RA=False, action_degrees=degrees, action_space=action_space, continuous=True)


if __name__ == "__main__": 

    Car()