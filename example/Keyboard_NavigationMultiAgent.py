import argparse
import gym
from gym_unrealcv.envs.wrappers import time_dilation, early_done, monitor, augmentation, configUE,agents
from pynput import keyboard
import time
import cv2
# Set UnrealEnv before running, for example:
#   Unix shell (bash/zsh):      export UnrealEnv=/path/to/UnrealEnv
#   Windows cmd.exe:            set UnrealEnv=C:\path\to\UnrealEnv
#   Windows PowerShell:         $env:UnrealEnv = "C:\path\to\UnrealEnv"
#   Python fallback:            os.environ.setdefault("UnrealEnv", "/path/to/UnrealEnv")

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
    if key_state['head_up']:
        action[1] = 1
    if key_state['head_down']:
        action[1] = 2

    action[0] = tuple(action[0])  # Convert inner list back to tuple
    action = tuple(action)  # Convert list back to tuple
    return action
def get_key_action_continuous():
    action = [0, 0]
    if key_state['i']:
        action[1] = 100
    if key_state['k']:
        action[1] = -100
    if key_state['j']:
        action[0] = -30
    if key_state['l']:
        action[0] = 30

    return action

def get_key_action_drone():
    action = [0, 0,0,0]
    if key_state['w']:
        action[0] = 1
    if key_state['s']:
        action[0] = -1
    if key_state['a']:
        action[3] = -1
    if key_state['d']:
        action[3] = 1
    if key_state['e']:
        action[2]=1
    if key_state['q']:
        action[2]=-1
    return action

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=None)
    # parser.add_argument("-e", "--env_id", nargs='?', default='UnrealTrack-track_train-ContinuousMask-v4',
    #                     help='Select the environment to run')
    parser.add_argument("-e", "--env_id", nargs='?', default='UnrealNavigationMulti-SuburbNeighborhood_Day-ContinuousColor-v0',
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
    env.unwrapped.agents_category=['player','drone'] #choose the agent type in the scene

    if int(args.time_dilation) > 0:  # -1 means no time_dilation
        env = time_dilation.TimeDilationWrapper(env, int(args.time_dilation))
    if int(args.early_done) > 0:  # -1 means no early_done
        env = early_done.EarlyDoneWrapper(env, int(args.early_done))
    if args.monitor:
        env = monitor.DisplayWrapper(env)
    env = augmentation.RandomPopulationWrapper(env, 2, 2, random_target=False)
    rewards = 0
    done = False
    Total_rewards = 0
    count_step = 0
    env.seed(int(args.seed))
    obs = env.reset()
    t0 = time.time()
    actions=[]
    agents_num = len(env.action_space)
    random_agents = [RandomAgent(env.action_space[i]) for i in range(agents_num)] # use random policy as the base policy for each agent
    print('''Use the "I", "J", "K", and "L" keys to control the first agent's movement, "Space" to jump, and "Up" and "Down" to adjust the agents view. (Double "Jump" will trigger the agent to climb)''')
    while True:

        actions = [random_agents[i].act(obs[i]) for i in range(agents_num)] #assign random policy for all agents
        actions[0] = get_key_action_continuous() # overwrite keyboard control policy for the first agent's movement (human character)
        # actions[1]=get_key_action_drone()# overwrite keyboard control policy for the second agent's movement (drone)
        obs, rewards, done, info = env.step(actions)
        for i in range(agents_num):
            cv2.imshow(f'Agent {i} observation',obs[i])
        cv2.waitKey(1)
        count_step += 1
        if done:
            if info['Success']:
                print('Success')
            else:
                print('Failed')
            fps = count_step / (time.time() - t0)
            print('Fps:' + str(fps))
            break
    env.close()