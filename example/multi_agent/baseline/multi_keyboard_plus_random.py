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
    'f': False,
    'h': False,
    'e': False,
    'ctrl': False,
    'space': False,
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
        if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
            key_state['ctrl'] = True
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
        if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
            key_state['ctrl'] = False
        if key == keyboard.Key.up:
            key_state['head_up'] = False
        if key == keyboard.Key.down:
            key_state['head_down'] = False
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()


def show_task_intro():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    map_img_path = os.path.join(project_root, 'doc', 'figs', 'navigation', 'map_2.png')
    target_img_path = os.path.join(project_root, 'doc', 'figs', 'navigation', 'target_2.png')

    print('\n=== NavigationMulti Task ===')
    print('Goal: the controlled agent navigates to the target object shown in the reference image.')
    print('Other agents keep random policy behavior.')
    print(f'Reference map: {map_img_path}')
    print(f'Reference target: {target_img_path}')
    print('Press Enter/Space to start, Esc to skip intro.')

    map_img = cv2.imread(map_img_path)
    target_img = cv2.imread(target_img_path)
    if map_img is None or target_img is None:
        print('Warning: intro images not found, skip image preview.')
        return

    target_h = 480
    map_w = max(1, int(map_img.shape[1] * (target_h / map_img.shape[0])))
    tgt_w = max(1, int(target_img.shape[1] * (target_h / target_img.shape[0])))
    map_rs = cv2.resize(map_img, (map_w, target_h), interpolation=cv2.INTER_AREA)
    tgt_rs = cv2.resize(target_img, (tgt_w, target_h), interpolation=cv2.INTER_AREA)
    panel = cv2.hconcat([map_rs, tgt_rs])
    cv2.putText(panel, 'Map (left) / Target (right)', (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

    window_name = 'NavigationMulti Intro'
    cv2.imshow(window_name, panel)
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in (13, 32, 27):  # Enter / Space / Esc
            break
    cv2.destroyWindow(window_name)


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
    if key_state['f']:
        action[2] = 3
    if key_state['h']:
        action[2] = 4
    if key_state['e']:
        action[2] = 5
    if key_state['ctrl']:
        action[2] = 2
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
    parser.add_argument("-e", "--env_id", nargs='?', default='UnrealNavigationMulti-SuburbNeighborhood_Day-MixedColor-v0',
                        help='Select the environment to run')
    parser.add_argument("-r", '--render', dest='render', action='store_true', help='show env using cv2')
    parser.add_argument("-s", '--seed', dest='seed', default=10, help='random seed')
    parser.add_argument("-t", '--time-dilation', dest='time_dilation', default=-1,
                        help='time_dilation to keep fps in simulator')
    parser.add_argument("-d", '--early-done', dest='early_done', default=-1, help='early_done when lost in n steps')
    parser.add_argument("-m", '--monitor', dest='monitor', action='store_true', help='auto_monitor')

    args = parser.parse_args()
    show_task_intro()
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
    print('Use I/J/K/L for first agent movement, F=open_door, H=enter/exit vehicle, Ctrl=crouch, E=pickup, Space=jump, Up/Down=view; other agents use random policy.')
    while True:

        actions = [random_agents[i].act(obs[i]) for i in range(agents_num)] # assign random policy for all agents
        actions[0] = get_key_action() # overwrite keyboard control for the first agent (human character, Mixed action)
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
