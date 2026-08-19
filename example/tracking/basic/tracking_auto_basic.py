"""
Entry point: multi-agent tracking demo.

Typical chain: ``gym.make`` → ``ConfigUEWrapper`` → optional wrappers →
``RandomPopulationWrapper``. On each ``reset``, the unwrapped env launches UE (once),
then ``set_population`` spawns agents from the Track JSON (e.g. ``Track/FlexibleRoom.json``)
using templates for empty ``name`` lists. ``env.unwrapped.agents_category`` (e.g. ``['player']``)
selects which agent type block in that JSON is used as the spawn template
(see ``UnrealCv_base._default_agent_template``).
"""
import argparse
import gym_unrealcv  # noqa: F401 — must load first to run register() in gym_unrealcv.__init__
import gym
from gym import wrappers
import cv2
import time
import numpy as np
from gym_unrealcv.envs.wrappers import time_dilation, early_done, monitor, agents, augmentation,configUE
from gym_unrealcv.envs.tracking.baseline import PoseTracker, Nav2GoalAgent
import os
# Optional local override: uncomment and set this path to run without defining
# UnrealEnv in the command line.
# os.environ['UnrealEnv'] = r'C:\path\to\UnrealEnv'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("-e", "--env_id", nargs='?', default='UnrealTrack-FlexibleRoom-ContinuousColor-v0',
                        help='Select the environment to run')
    parser.add_argument("-r", '--render', dest='render', action='store_true', help='show env using cv2')
    parser.add_argument("-s", '--seed', dest='seed', default=0, help='random seed')
    parser.add_argument("-t", '--time_dilation', dest='time_dilation', default=-1, help='time_dilation to keep fps in simulator')
    parser.add_argument("-d", '--early_done', dest='early_done', default=-1, help='early_done when lost in n steps')
    parser.add_argument("-m", '--monitor', dest='monitor', action='store_true', help='auto_monitor')

    args = parser.parse_args()
    env = gym.make(args.env_id)
    env = configUE.ConfigUEWrapper(env, offscreen=False, resolution=(240, 240))
    # Which JSON agent template to spawn (must match a key under ``agents`` in the env JSON).
    env.unwrapped.agents_category = ['player']

    if int(args.time_dilation) > 0:  # -1 means no time_dilation
        env = time_dilation.TimeDilationWrapper(env, args.time_dilation)
    if int(args.early_done) > 0:  # -1 means no early_done
        env = early_done.EarlyDoneWrapper(env, args.early_done)
    if args.monitor:
        env = monitor.DisplayWrapper(env)

    env = augmentation.RandomPopulationWrapper(env, 2, 2, random_target=False)
    # env = agents.NavAgents(env, mask_agent=True)
    episode_count = 1000
    rewards = 0
    done = False

    Total_rewards = 0
    env.seed(int(args.seed))
    try:
        for eps in range(1, episode_count):
            obs = env.reset()
            agents_num = len(env.action_space)
            tracker_id = env.unwrapped.tracker_id
            target_id = env.unwrapped.target_id
            # trackers = [PoseTracker(env.action_space[i], env.unwrapped.exp_distance) for i in range(agents_num) ]
            trackers = [PoseTracker(env.action_space[i], env.unwrapped.reward_params['exp_distance']) for i in range(agents_num) if i%2==1]
            targets = [Nav2GoalAgent(env.action_space[i], env.unwrapped.reset_area, max_len=100) for i in range(agents_num) if i%2==0]
            count_step = 0
            t0 = time.time()
            agents_num = len(obs)
            C_rewards = np.zeros(agents_num)
            print('eps:', eps, 'agents_num:', agents_num)
            while True:
                obj_poses = env.unwrapped.obj_poses
                # actions = [trackers[i].act(obj_poses[i], obj_poses[target_id]) for i in range(agents_num)]
                actions = []
                for i in range(agents_num):
                    if i%2==0:
                        action = targets[i//2].act(obj_poses[i])
                    else:
                        action = trackers[i//2].act(obj_poses[i], obj_poses[i-1])
                    actions.append(action)
                obs, rewards, done, _ = env.step(actions)
                C_rewards += rewards
                count_step += 1
                if args.render:
                    img = env.render(mode='rgb_array')
                    #  img = img[..., ::-1]  # bgr->rgb
                cv2.imshow('show', obs[0][:,:,:3])
                cv2.waitKey(1)
                if done:
                    fps = count_step/(time.time() - t0)
                    Total_rewards += C_rewards[0]
                    print('Fps:' + str(fps), 'R:'+str(C_rewards), 'R_ave:'+str(Total_rewards/eps))
                    break

        # Close the env and write monitor result info to disk
        print('Finished')
        env.close()
    except KeyboardInterrupt:
        print('exiting')
        env.close()


