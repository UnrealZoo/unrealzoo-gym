from gym_unrealcv.envs.base_env import UnrealCv_base
import numpy as np
import random
'''
Tasks: The task is to make the agents find the target(injured person) and rescue him. 
The agents are allowed to communicate with others.
The agents observe the environment with their own camera.
The agents are rewarded based on the distance and orientation to the target.
The episode ends when the agents reach the target(injured agent) or the maximum steps are reached.
'''

class Rescue(UnrealCv_base):
    def __init__(self,
                 env_file,  # the setting file to define the task
                 task_file=None,  # the file to define the task TODO: use this file to config task specific parameters
                 action_type='Discrete',  # 'discrete', 'continuous'
                 observation_type='Color',  # 'color', 'depth', 'rgbd', 'Gray'
                 resolution=(160, 160)
                 ):
        super(Rescue, self).__init__(setting_file=env_file,  # the setting file to define the task
                                         action_type=action_type,  # 'discrete', 'continuous'
                                         observation_type=observation_type,  # 'color', 'depth', 'rgbd', 'Gray'
                                         resolution=resolution)
        self.task_config = self.load_task_setting(task_file)
        self.count_reach = 0
        self.max_reach_steps = self.task_config.get('max_reach_steps', 5)
        self.distance_threshold = self.task_config.get('distance_threshold', 200)
        self.agents_category = ['player']
        self.injured_agent = None
        self.injured_agent_name = self.task_config.get('injured_agent_name', 'injured_person')
        self.reward_type = self.task_config.get('reward_type', 'shared')  # 'sparse', 'shared', 'individual'
        ## TODO: add trigger action

    def step(self, action):
        obs, rewards, done, info = super(Rescue, self).step(action)
        # compute the useful metrics for rewards and done condition
        metrics = self.rescue_metrics(info['Pose'], self.target_pose)
        rewards = self.reward(metrics)
        info['Reward'] = rewards
        # done condition
        if metrics['reach'].max() == 1:
            self.count_reach += 1
        if self.count_reach > self.max_reach_steps:
            info['Done'] = True
            done = True
        return obs, rewards, done, info

    def reset(self):
        # initialize the environment
        states = super(Rescue, self).reset()
        super(Rescue, self).random_app()
        if self.injured_agent is None:
            # add the injured person
            class_name = 'bp_character_C'
            loc = random.choice(self.safe_start)
            self.injured_agent = self.injured_agent_name
            self.unrealcv.new_obj(class_name, self.injured_agent, loc)
            self.unrealcv.set_phy(self.injured_agent, 0)
        else:
            self.unrealcv.set_standup(self.injured_agent)
            self.unrealcv.set_obj_location(self.injured_agent, random.choice(self.safe_start))
        candidate_direction = [[100, 0], [-100, 0], [0, 100], [0, -100]]
        self.unrealcv.set_liedown(self.injured_agent, random.choice(candidate_direction))
        self.target_pose = self.unrealcv.get_obj_pose(self.injured_agent)
        self.count_reach = 0
        return states

    def reward(self, metrics):
        # individual reward
        if 'individual' in self.reward_type:
            if 'sparse' in self.reward_type:
                rewards = metrics['reach']  # only the agent who reach the target get the reward
            else:
                rewards = 1 - metrics['dis_each']/self.distance_threshold - np.fabs(metrics['ori_each'])/180 + metrics['reach']
        elif 'shared' in self.reward_type:
            if 'sparse' in self.reward_type:
                rewards = metrics['reach'].max()
            else:
                rewards = 1 - metrics['dis_min']/self.distance_threshold + metrics['reach'].max()
        else:
            raise ValueError('reward type is not defined')
        return rewards

    def rescue_metrics(self, objs_pose, target_loc):
        # compute the relative relation (distance, collision) among agents for rewards and evaluation metrics
        info = dict()
        relative_pose = []
        for obj_pos in objs_pose:
            obs, distance, direction = self.get_relative(obj_pos, target_loc)
            relative_pose.append(np.array([distance, direction]))
        relative_pose = np.array(relative_pose)
        relative_dis = relative_pose[:, 0]
        relative_ori = relative_pose[:, 1]
        reach_mat = np.zeros_like(relative_dis)
        reach_mat[np.where(relative_dis < self.distance_threshold)] = 1
        reach_mat[np.where(np.fabs(relative_ori) > 45)] = 0  # collision should be at the front view
        info['reach'] = reach_mat
        info['dis_min'] = relative_dis.min(-1)  # minimal distance from agents to target
        info['dis_each'] = relative_dis  # distance from agents to target
        info['ori_each'] = relative_ori  # orientation from agents to target
        return info