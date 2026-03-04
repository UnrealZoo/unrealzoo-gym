# place multiple agent in the environment, each agent share the same action space
# each agent perform randome action in the environment
import os
import argparse
import gym_unrealcv
from gym_unrealcv._gym_compat import gym
import cv2
import time
import numpy as np
from gym_unrealcv.envs.wrappers import time_dilation, early_done, monitor, agents, augmentation, configUE

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space
        self.count_steps = 0
        self.action = self.action_space.sample()

    def act(self, observation, keep_steps=10):
        self.count_steps += 1
        if self.count_steps > keep_steps:
            self.action = self.action_space.sample()
            self.count_steps = 0
        else:
            return self.action
        return self.action

    def reset(self):
        self.action = self.action_space.sample()
        self.count_steps = 0

import os
os.environ['UnrealEnv']='/home/wuk/UnrealEnv'
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("-e", "--env_id", nargs='?', default='UnrealTrack-MiddleEast-ContinuousColor-v0',
                        help='Select the environment to run')
    parser.add_argument("-r", '--render', dest='render', action='store_true', help='show env using cv2')
    parser.add_argument("-s", '--seed', dest='seed', type=int, default=0, help='random seed')
    parser.add_argument("-t", '--time-dilation', dest='time_dilation', type=int, default=10, help='time_dilation to keep fps in simulator')
    parser.add_argument("-n", '--nav-agent', dest='nav_agent', action='store_true', help='use nav agent to control the agents')
    parser.add_argument("-d", '--early-done', dest='early_done', type=int, default=-1, help='early_done when lost in n steps')
    parser.add_argument("-m", '--monitor', dest='monitor', action='store_true', help='auto_monitor')

    args = parser.parse_args()
    env = gym.make(args.env_id)
    env = configUE.ConfigUEWrapper(env, offscreen=False,resolution=(240,240))
    env.unwrapped.agents_category=['player'] #choose the agent type in the scene

    if args.time_dilation > 0:  # -1 means no time_dilation
        env = time_dilation.TimeDilationWrapper(env, args.time_dilation)
    if args.early_done > 0:  # -1 means no early_done
        env = early_done.EarlyDoneWrapper(env, args.early_done)
    if args.monitor:
        env = monitor.DisplayWrapper(env)

    env = augmentation.RandomPopulationWrapper(env, 2, 2, random_target=False)
    if args.nav_agent:
        env = agents.NavAgents(env, mask_agent=False)
    episode_count = 100
    rewards = 0
    done = False

    Total_rewards = 0.0
    env.seed(args.seed)
    for eps in range(1, episode_count):
        obs = env.reset()
        agents_num = len(env.action_space)
        random_agents = [RandomAgent(env.action_space[i]) for i in range(agents_num)]  # reset agents
        count_step = 0
        t0 = time.time()
        agents_num = len(obs)
        C_rewards = np.zeros(agents_num)
        while True:
            actions = [random_agents[i].act(obs[i]) for i in range(agents_num)]
            obs, rewards, done, info = env.step(actions)
            C_rewards += rewards
            count_step += 1
            if args.render:
                img = env.render(mode='rgb_array')
                #  img = img[..., ::-1]  # bgr->rgb
                cv2.imshow('show', img)
                cv2.waitKey(1)
            if done:
                fps = count_step/(time.time() - t0)
                Total_rewards += float(C_rewards[0])
                print ('Fps:' + str(fps), 'R:'+str(C_rewards), 'R_ave:'+str(Total_rewards/eps))
                break

    # Close the env and write monitor result info to disk
    print('Finished')
    env.close()



