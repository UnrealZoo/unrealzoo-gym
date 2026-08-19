import argparse
import gym
from gym_unrealcv.envs.wrappers import time_dilation, early_done, monitor, augmentation, configUE,agents
from pynput import keyboard
import time
import cv2
import os
# Optional local override: uncomment and set this path to run without defining
# UnrealEnv in the command line.
# os.environ['UnrealEnv'] = r'C:\path\to\UnrealEnv'

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation):
        # return 2
        return self.action_space.sample()


key_state = {
    'i': False,
    'j': False,
    'k': False,
    'l': False,
    'space': False,
    '1': False,
    '2': False,
'3': False,
    'head_up': False,
    'head_down': False
}

def on_press(key):
    try:
        if key.char in key_state:
            key_state[key.char] = True
    except AttributeError:
        if key == keyboard.Key.space:
            key_state['space'] = True
        if key == keyboard.Key.up:
            key_state['head_up'] = True
        if key == keyboard.Key.down:
            key_state['head_down'] = True


def on_release(key):
    try:
        if key.char in key_state:
            key_state[key.char] = False
    except AttributeError:
        if key == keyboard.Key.space:
            key_state['space'] = False
        if key == keyboard.Key.up:
            key_state['head_up'] = False
        if key == keyboard.Key.down:
            key_state['head_down'] = False
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()
def get_key_action():
    action = ([0, 0], 0, 0)
    action = list(action)  # Convert tuple to list for modification
    action[0] = list(action[0])  # Convert inner tuple to list for modification

    if key_state['i']:
        action[0][1] = 100
    if key_state['k']:
        action[0][1] = -100
    if key_state['j']:
        action[0][0] = -30
    if key_state['l']:
        action[0][0] = 30
    if key_state['space']:
        action[2] = 1
    if key_state['1']:
        action[2] = 3
    if key_state['2']:
        action[2] = 4
    if key_state['3']:
        action[2] = 5
    if key_state['head_up']:
        action[1] = 1
    if key_state['head_down']:
        action[1] = 2

    action[0] = tuple(action[0])  # Convert inner list back to tuple
    action = tuple(action)  # Convert list back to tuple
    return action


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=None)
    # parser.add_argument("-e", "--env_id", nargs='?', default='UnrealTrack-track_train-ContinuousMask-v4',
    #                     help='Select the environment to run')
    parser.add_argument("-e", "--env_id", nargs='?', default='UnrealTrack-DowntownWest-MixedColor-v0',
                        help='Select the environment to run')
    parser.add_argument("-r", '--render', dest='render', action='store_true', help='show env using cv2')
    parser.add_argument("-s", '--seed', dest='seed', default=10, help='random seed')
    parser.add_argument("-t", '--time-dilation', dest='time_dilation', default=-1,
                        help='time_dilation to keep fps in simulator')
    parser.add_argument("-d", '--early-done', dest='early_done', default=-1, help='early_done when lost in n steps')
    parser.add_argument("-m", '--monitor', dest='monitor', action='store_true', help='auto_monitor')

    args = parser.parse_args()
    env = gym.make(args.env_id)
    env = configUE.ConfigUEWrapper(env, offscreen=False, resolution=(240, 240))
    env.unwrapped.agents_category=['animal'] #choose the agent type in the scene

    if int(args.time_dilation) > 0:  # -1 means no time_dilation
        env = time_dilation.TimeDilationWrapper(env, int(args.time_dilation))
    if int(args.early_done) > 0:  # -1 means no early_done
        env = early_done.EarlyDoneWrapper(env, int(args.early_done))
    if args.monitor:
        env = monitor.DisplayWrapper(env)
    env = augmentation.RandomPopulationWrapper(env, 2, 2, random_target=False)
    env = agents.NavAgents(env, mask_agent=False)
    agent = RandomAgent(env.action_space[0])
    rewards = 0
    done = False
    Total_rewards = 0
    count_step = 0
    env.seed(int(args.seed))
    obs = env.reset()
    t0 = time.time()
    print('Use the "I", "J", "K", and "L" keys to control the agent  movement.')
    while True:
        action = get_key_action()
        obs, rewards, done, info = env.step([action])
        cv2.imshow('obs',obs[0])
        cv2.waitKey(1)
        count_step += 1
        if done:
            fps = count_step / (time.time() - t0)
            print('Success')
            print('Fps:' + str(fps))
            break
    env.close()
