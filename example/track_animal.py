import argparse
import gym_unrealcv
import gym
from gym import wrappers
import cv2
import time
import numpy as np
from gym_unrealcv.envs.wrappers import time_dilation, early_done, monitor, agents, augmentation,configUE
from gym_unrealcv.envs.tracking.baseline import PoseTracker, Nav2GoalAgent
import os
os.environ['UnrealEnv']='/path/to/your/UnrealEnv'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("-e", "--env_id", nargs='?', default='UnrealTrack-FlexibleRoom-ContinuousColor-v0',
                        help='Select the environment to run')
    parser.add_argument("-r", '--render', dest='render', action='store_true', help='show env using cv2')
    parser.add_argument("-s", '--seed', dest='seed', default=0, help='random seed')
    parser.add_argument("-t", '--time_dilation', dest='time_dilation', default=10, help='time_dilation to keep fps in simulator')
    parser.add_argument("-d", '--early_done', dest='early_done', default=-1, help='early_done when lost in n steps')
    parser.add_argument("-m", '--monitor', dest='monitor', action='store_true', help='auto_monitor')

    args = parser.parse_args()
    env = gym.make(args.env_id)
    env.unwrapped.agents_category=['player','animal'] #choose the agent type in the scene
    env = configUE.ConfigUEWrapper(env, offscreen=False, resolution=(240, 240))
    # env.unwrapped.agents_category=['player'] #choose the agent type in the scene
    if int(args.time_dilation) > 0:  # -1 means no time_dilation
        env = time_dilation.TimeDilationWrapper(env, args.time_dilation)
    if int(args.early_done) > 0:  # -1 means no early_done
        env = early_done.EarlyDoneWrapper(env, args.early_done)
    if args.monitor:
        env = monitor.DisplayWrapper(env)

    env = augmentation.RandomPopulationWrapper(env, 2, 2, random_target=False)
    env = agents.NavAgents(env, mask_agent=True) #use internal nav system for target moving
    episode_count = 100
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

            # 设置智能体速度
            tracker_name = env.unwrapped.player_list[tracker_id]
            target_name = env.unwrapped.player_list[target_id]
            env.unwrapped.unrealcv.set_max_speed(target_name, 100)   # target (animal) 速度

            # 使用环境自动设置的 tracker_id 和 target_id
            # tracker_id = 0 (player 追踪者), target_id = 1 (animal 被追踪者)
            print(f'Tracker ID: {tracker_id} ({env.unwrapped.agents_category[tracker_id]}), Target ID: {target_id} ({env.unwrapped.agents_category[target_id]})')
            print(f'Tracker speed: 100, Target speed: 100')

            # 为tracker智能体创建控制器, target 采用内置的导航控制器
            trackers = PoseTracker(env.action_space[0])

            count_step = 0
            t0 = time.time()
            agents_num = len(obs)
            C_rewards = np.zeros(agents_num)
            print('eps:', eps, 'agents_num:', agents_num)
            while True:

                obj_poses = env.unwrapped.obj_poses
                action = trackers.act(obj_poses[0], obj_poses[1])
                obs, rewards, done, _ = env.step([action])
                C_rewards += rewards
                count_step += 1
                if args.render:
                    img = env.render(mode='rgb_array')
                    #  img = img[..., ::-1]  # bgr->rgb
                # 显示 tracker（追踪者）的视角
                cv2.imshow('show', obs[tracker_id])
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



