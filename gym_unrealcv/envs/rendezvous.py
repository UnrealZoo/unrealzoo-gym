from gym_unrealcv.envs.base_env import UnrealCv_base
import numpy as np
'''
Tasks: The task is to make the agents meet at a rendezvous point. 
The agents are allowed to communicate with others to reach consensus (where to meet).
The agents observe the environment with their own camera.
The agents are rewarded based on the distance to the other agents.
The episode ends when the agents meet or the maximum steps are reached.
'''

class Rendezvous(UnrealCv_base):
    def __init__(self,
                 env_file,  # the setting file to define the task
                 task_file=None,  # the file to define the task TODO: use this file to config task specific parameters
                 action_type='Discrete',  # 'discrete', 'continuous'
                 observation_type='Color',  # 'color', 'depth', 'rgbd', 'Gray'
                 resolution=(160, 160),
                 reset_type=0
                 ):
        super(Rendezvous, self).__init__(setting_file=env_file,  # the setting file to define the task
                                         action_type=action_type,  # 'discrete', 'continuous'
                                         observation_type=observation_type,  # 'color', 'depth', 'rgbd', 'Gray'
                                         resolution=resolution,
                                         reset_type=reset_type)
        self.task_config = self.load_task_setting(task_file)
        self.count_meet = 0
        self.max_meet_steps = self.task_config.get('max_meet_steps', 20)
        self.distance_threshold = self.task_config.get('distance_threshold', 200)
        self.agents_category = ['player']

    def step(self, action):
        obs, rewards, done, info = super(Rendezvous, self).step(action)
        # compute the useful metrics for rewards and done condition
        metrics = self.rendezvous_metrics(info['Relative_Pose'])
        rewards = self.reward(metrics)
        info['Reward'] = rewards
        # done condition
        if metrics['dis_ave'].mean() < self.distance_threshold:
            self.count_meet += 1
        if self.count_meet > self.max_meet_steps:
            info['Done'] = True
            done = True
        return obs, rewards, done, info

    def reset(self):
        states = super(Rendezvous, self).reset()
        super(Rendezvous, self).random_app()
        self.count_meet = 0
        return states

    def reward(self, metrics):
        rewards = 1 - metrics['dis_ave']/self.distance_threshold
        return rewards

    def rendezvous_metrics(self, relative_pose):
        # compute the relative relation (distance, collision) among agents for rewards and evaluation metrics
        info = dict()
        relative_dis = relative_pose[:, :, 0]
        relative_ori = relative_pose[:, :, 1]
        collision_mat = np.zeros_like(relative_dis)
        collision_mat[np.where(relative_dis < 100)] = 1
        collision_mat[np.where(np.fabs(relative_ori) > 45)] = 0  # collision should be at the front view
        info['collision'] = collision_mat
        info['dis_ave'] = relative_dis.mean(-1)  # average distance between agents, e.g., i to the others in average
        return info