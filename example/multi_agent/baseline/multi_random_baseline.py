# place multiple agent in the environment, each agent share the same action space
# each agent perform randome action in the environment

import argparse
import gym_unrealcv
import gym
from gym import wrappers
import cv2
import time
import numpy as np
import os
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("-e", "--env_id", nargs='?', default='UnrealTrack-FlexibleRoom-ContinuousColor-v0',
                        help='Select the environment to run')
    parser.add_argument("-r", '--render', dest='render', action='store_true', help='show env using cv2')
    parser.add_argument("-s", '--seed', dest='seed', default=0, help='random seed')
    parser.add_argument("-t", '--time-dilation', dest='time_dilation', default=-1, help='time_dilation to keep fps in simulator')
    parser.add_argument("-n", '--nav-agent', dest='nav_agent', action='store_true', help='use nav agent to control the agents')
    parser.add_argument("-d", '--early-done', dest='early_done', default=-1, help='early_done when lost in n steps')
    parser.add_argument("-m", '--monitor', dest='monitor', action='store_true', help='auto_monitor')
    parser.add_argument('--unreal-env', default=None,
                        help='UnrealEnv root. Example on Windows: J:\\UnrealEnv')
    parser.add_argument('--episodes', type=int, default=99,
                        help='number of episodes to run (default keeps the original 99)')
    parser.add_argument('--num-agents', type=int, default=10,
                        help='fixed random-agent population (default keeps the original 10)')
    parser.add_argument('--max-steps', type=int, default=0,
                        help='stop each episode after N steps; 0 uses the environment limit')
    parser.add_argument('--no-display', action='store_true',
                        help='do not open OpenCV windows (UE rendering and observations stay enabled)')
    parser.add_argument('--save-first-frame', default=None,
                        help='directory in which to save the first color/mask observation')

    args = parser.parse_args()
    if args.unreal_env:
        os.environ['UnrealEnv'] = args.unreal_env
    env = gym.make(args.env_id)
    env = configUE.ConfigUEWrapper(env, offscreen=False,resolution=(640,640))
    env.unwrapped.agents_category=['player'] #choose the agent type in the scene

    if int(args.time_dilation) > 0:  # -1 means no time_dilation
        env = time_dilation.TimeDilationWrapper(env, int(args.time_dilation))
    if int(args.early_done) > 0:  # -1 means no early_done
        env = early_done.EarlyDoneWrapper(env, int(args.early_done))
    if args.monitor:
        env = monitor.DisplayWrapper(env)

    env = augmentation.RandomPopulationWrapper(env, args.num_agents, args.num_agents, random_target=False)
    if args.nav_agent:
        env = agents.NavAgents(env, mask_agent=False)
    episode_count = args.episodes
    rewards = 0
    done = False

    Total_rewards = 0
    env.seed(int(args.seed))
    try:
        for eps in range(1, episode_count + 1):
            obs = env.reset()
            print('episode:', eps, 'observation shape:', obs[0].shape, flush=True)
            if args.save_first_frame and eps == 1:
                os.makedirs(args.save_first_frame, exist_ok=True)
                cv2.imwrite(os.path.join(args.save_first_frame, 'color.png'), obs[0][:, :, :3])
                if obs[0].shape[2] >= 6:
                    cv2.imwrite(os.path.join(args.save_first_frame, 'mask.png'), obs[0][:, :, 3:6])
                elif 'Mask' in args.env_id:
                    cv2.imwrite(os.path.join(args.save_first_frame, 'mask.png'), obs[0][:, :, :3])
            agents_num = len(env.action_space)
            agents = [RandomAgent(env.action_space[i]) for i in range(agents_num)]  # reset agents
            count_step = 0
            t0 = time.time()
            agents_num = len(obs)
            C_rewards = np.zeros(agents_num)
            while True:
                actions = [agents[i].act(obs[i]) for i in range(agents_num)]
                obs, rewards, done, info = env.step(actions)
                C_rewards += rewards
                count_step += 1
                if args.render and not args.no_display:
                    img = env.render(mode='rgb_array')
                    #  img = img[..., ::-1]  # bgr->rgb
                    cv2.imshow('show', img)
                    cv2.waitKey(1)
                if not args.no_display:
                    cv2.imshow('color', obs[0][:, :, :3])
                    if obs[0].shape[2] >= 6:
                        cv2.imshow('mask', obs[0][:, :, 3:6])
                    cv2.waitKey(1)
                if done or (args.max_steps > 0 and count_step >= args.max_steps):
                    fps = count_step/(time.time() - t0)
                    Total_rewards += C_rewards[0]
                    print ('Fps:' + str(fps), 'R:'+str(C_rewards), 'R_ave:'+str(Total_rewards/eps))
                    break

        print('Finished')
    finally:
        # Close the env even when UE crashes or the socket is interrupted.
        env.close()



