import time
import gym_unrealcv
from gym_unrealcv.envs.base_env import UnrealCv_base
import numpy as np
import cv2
import random
import os
'''
Tasks: The task is to make the agents find the injured person and rescue him. 
The agents are allowed to communicate with others to plan.
The agents observe the environment with their own camera.
The agents are rewarded based on the distance to the other agents.
The episode ends when the agents meet or the maximum steps are reached.
'''

class Track(UnrealCv_base):
    def __init__(self,
                 env_file,  # the setting file to define the task
                 task_file=None,  # the file to define the task TODO: use this file to config task specific parameters
                 action_type='Discrete',  # 'discrete', 'continuous'
                 observation_type='Color',  # 'color', 'depth', 'rgbd', 'Gray'
                 resolution=(160, 160),
                 reset_type = 0,
                 num_agents=2,
                 ):
        super(Track, self).__init__(setting_file=env_file,  # the setting file to define the task
                                         action_type=action_type,  # 'discrete', 'continuous'
                                         observation_type=observation_type,  # 'color', 'depth', 'rgbd', 'Gray'
                                         resolution=resolution,
                                    reset_type=reset_type,
                                    num_agents=num_agents)
        self.count_lost = 0
        self.max_lost_steps = 20
        self.agents_category = ['player']
        self.reward_type = 'dense'  # 'dense', 'sparse'
        self.reward_params = {
            "min_distance": 100,
            "max_direction": 60,
            "max_distance": 750,
            "exp_distance": 250,
            "exp_angle": 0
        }
        self.distance_threshold = self.reward_params["min_distance"]  # distance threshold for collision
        self.tracker_id = self.protagonist_id
        self.target_id = self.protagonist_id+1

    def step(self, action):
        obs, rewards, done, info = super(Track, self).step(action)
        relative_pose = info['Relative_Pose']
        # compute the useful metrics for rewards and done condition
        metrics, score4tracker = self.track_metrics(relative_pose, self.tracker_id, self.target_id)

        # prepare the info
        info['Distance'], info['Direction'] = relative_pose[self.tracker_id][self.target_id]
        info['Relative_Pose'] = relative_pose
        rewards = self.get_rewards(score4tracker, metrics, self.tracker_id, self.target_id)

        info['Reward'] = rewards
        info['metrics'] = metrics

        return obs, rewards, done, info

    def reset(self):
        # initialize the environment
        observations = super(Track, self).reset()

        target_pos = self.unrealcv.get_obj_location(self.player_list[self.target_id])
        # print(target_pos)
        self.unrealcv.nav_to_goal(self.player_list[self.target_id], target_pos)
        time.sleep(1)
        super(Track, self).random_app()
        object_list = self.unrealcv.get_objects()
        ############
        for obj in object_list: #binary mask configure
            if obj == self.player_list[self.target_id]:
                self.unrealcv.set_obj_color(obj, (255, 255, 255))
            else:
                if 'floor' in obj.lower() or 'wall' in obj.lower():
                    if random.random()<0.5:
                        self.unrealcv.set_obj_color(obj, (0, 0, 0))
                    else:
                        random_color = color = np.random.randint(100, 255, 3)
                        self.unrealcv.set_obj_color(obj, random_color)
            #     if random.random() < 0.9:
            #         if obj not in self.objects_list and obj not in self.player_list:
            #             try:
            #                 self.unrealcv.set_obj_color(obj, (0, 0, 0))
            #             except:
            #                 pass
            #     else:
            #         random_color=color = np.random.randint(100, 255, 3)
            #         if obj not in self.objects_list and obj not in self.player_list:
            #             try:
            #                 self.unrealcv.set_obj_color(obj, random_color)
            #             except:
            #                 pass
        ##############
        time.sleep(1)
        target_pos = self.unrealcv.get_obj_location(self.player_list[self.target_id])
        # initialize the tracker
        cam_pos_exp, yaw_exp= self.get_tracker_init_point(target_pos, self.reward_params["exp_distance"])
        # set tracker location
        tracker_name = self.player_list[self.tracker_id]
        self.unrealcv.set_obj_location(tracker_name, cam_pos_exp)
        self.unrealcv.set_obj_rotation(tracker_name, [0, yaw_exp, 0])
        # reset if cannot see the target at initial frame
        # try:
        #     while self.unwrapped.unrealcv.check_visibility(self.cam_list[self.tracker_id],self.player_list[self.target_id]) == 0 :
        #         target_locations = self.sample_init_pose()
        #         self.unrealcv.set_obj_location(self.player_list[self.target_id], target_locations[0])
        #         self.unrealcv.set_cam(self.player_list[self.target_id],
        #                               self.agents[self.player_list[self.target_id]]['relative_location'],
        #                               self.agents[self.player_list[self.target_id]]['relative_rotation'])
        #         target_pos = self.unrealcv.get_obj_location(self.player_list[self.target_id])
        #         # initialize the tracker
        #         cam_pos_exp, yaw_exp = self.get_tracker_init_point(target_pos, self.reward_params["exp_distance"])
        #         # set tracker location
        #         tracker_name = self.player_list[self.tracker_id]
        #         self.unrealcv.set_obj_location(tracker_name, cam_pos_exp)
        #         self.unrealcv.set_obj_rotation(tracker_name, [0, yaw_exp, 0])
        #         time.sleep(1)
        # except:
        #     pass

        # update the observation
        observations, self.obj_poses, self.img_show = self.update_observation(self.player_list, self.cam_list, self.cam_flag, self.observation_type)
        self.count_lost = 0
        return observations

    def track_metrics(self, relative_pose, tracker_id, target_id):
        # compute the relative relation (collision, in-the-view, misleading) among agents for rewards and evaluation metrics
        info = dict()
        relative_dis = relative_pose[:, :, 0]
        relative_ori = relative_pose[:, :, 1]
        collision_mat = np.zeros_like(relative_dis)
        collision_mat[np.where(relative_dis < 100)] = 1
        collision_mat[np.where(np.fabs(relative_ori) > 45)] = 0  # collision should be at the front view
        info['collision'] = collision_mat

        info['dis_ave'] = relative_dis.mean() # average distance among players, regard as a kind of density metric

        # if in the tracker's view
        view_mat = np.zeros_like(relative_ori)
        view_mat[np.where(np.fabs(relative_ori) < 45)] = 1
        view_mat[np.where(relative_dis > self.reward_params['max_distance'])] = 0
        view_mat_tracker = view_mat[tracker_id]
        # how many distractors are observed
        info['d_in'] = view_mat_tracker.sum() - view_mat_tracker[target_id] - view_mat_tracker[tracker_id]  # distractor in the observable area
        info['target_viewed'] = view_mat_tracker[target_id]  # target in the observable area

        # detect target mask to determine if in the view (not work for some environment, which cannot rendering mask, like industrialArea)
        target_percent = self.unwrapped.unrealcv.check_visibility(self.cam_list[self.tracker_id],
                                                                  self.player_list[self.target_id])
        info['target_viewed'] = int(target_percent > 0 and view_mat_tracker[target_id])

        if target_percent <= 0:
            self.count_lost += 1
        else:
            self.count_lost = 0

        relative_oir_norm = np.fabs(relative_ori-self.reward_params['exp_angle']) / 45.0
        relation_norm = np.fabs(relative_dis - self.reward_params['exp_distance'])/self.reward_params['max_distance'] + relative_oir_norm
        reward_tracker = 1 - relation_norm[0]  # measuring the quality among tracker to others
        info['tracked_id'] = np.argmax(reward_tracker)  # which one is tracked
        info['perfect'] = info['target_viewed'] * (info['d_in'] == 0) * (reward_tracker[1] > 0.5)
        info['mislead'] = 0
        if info['tracked_id'] > 1 and reward_tracker[info['tracked_id']] > 0.5: # only when target is far away to the center and distracotr is close
            advantage = reward_tracker[info['tracked_id']] - reward_tracker[1]
            if advantage > 1:
                info['mislead'] = info['tracked_id']

        return info, reward_tracker

    def get_rewards(self, score4tracker, metrics, tracker_id, target_id):
        rewards = []
        mask = np.ones(metrics['collision'][tracker_id].shape, dtype=bool)
        mask[tracker_id] = False
        tracker_collision = metrics['collision'][tracker_id]
        if self.reward_type == 'dense':
            r_tracker = score4tracker[target_id] - np.max(tracker_collision[mask])  #
            r_target = -score4tracker[target_id]
        elif self.reward_type == 'sparse':
            r_tracker = 1 if metrics['perfect'] > 0 else -1
            r_target = -r_tracker

        for i in range(len(self.player_list)):
            if i == tracker_id:
                rewards.append(r_tracker)
            elif i == target_id:  # target, try to run away
                rewards.append(r_target - tracker_collision[i])
            else:  # distractors, try to mislead tracker, and improve the target's reward.
                r_d = r_target + score4tracker[i]  # try to appear in the tracker's view
                r_d -= tracker_collision[i]
                if 'sparse' in self.reward_type:
                    r_d = 1 if r_d > 0 else -1
                rewards.append(r_d)
        return np.array(rewards)

    def get_tracker_init_point(self, target_pos, distance, direction=None):
        if direction is None:
            direction = 2 * np.pi * np.random.sample(1)
        else:
            direction = direction % (2 * np.pi)

        dx = float(distance * np.cos(direction))
        dy = float(distance * np.sin(direction))
        x = dx + target_pos[0]
        y = dy + target_pos[1]
        z = target_pos[2]
        cam_pos_exp = [x, y, z]
        yaw = float(direction / np.pi * 180 - 180)

        return [cam_pos_exp, yaw]

    def check_visibility(self, cam_id):
        mask = self.unrealcv.get_image(cam_id, 'object_mask', 'bmp')
        mask, bbox = self.unrealcv.get_bbox(mask, self.player_list[self.target_id], normalize=False)
        mask_percent = mask.sum()/(self.resolution[0] * self.resolution[1])
        return mask_percent

