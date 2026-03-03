import numpy as np
from gym_unrealcv.envs.base_env import UnrealCv_base
from gym_unrealcv.envs.utils import misc, reward

COLLISION_THRESHOLD = 10
SUCCESS_DISTANCE = 300
SUCCESS_DIRECTION = 10
SUCCESS_REWARD = 100
COLLISION_REWARD = -1

'''
It is a general env for navigating to a target object.

State : raw color image and depth (640x480) 
Action:  (linear velocity ,angle velocity , trigger) 
Done : Collision or get target place or False trigger three times.
Task: Learn to avoid obstacle and search for a target object in a room, 
      you can select the target name according to the Recommend object list in setting files

'''


class Navigation(UnrealCv_base):
    def __init__(self,
                 env_file,  # the setting file to define the task
                 task_file=None,  # the file to define the task
                 action_type='Discrete',  # 'discrete', 'continuous'
                 observation_type='Color',  # 'color', 'depth', 'rgbd', 'Gray'
                 resolution=(160, 160),
                 reset_type=0
                 ):
        super(Navigation, self).__init__(setting_file=env_file,  # the setting file to define the task
                                    action_type=action_type,  # 'discrete', 'continuous'
                                    observation_type=observation_type,  # 'color', 'depth', 'rgbd', 'Gray'
                                    resolution=resolution,
                                    reset_type=reset_type)


        # self.cam_id = self.setting['cam_id']
        self.target_list =self.env_configs['targets']['Point']

        self.player = self.player_list

        self.observation_type = observation_type
        # assert self.observation_type == 'Color' or self.observation_type == 'Depth' or self.observation_type == 'Rgbd' or self.observation_type == 'Mask'
        # self.observation_space = self.unrealcv.define_observation(self.cam_id, self.observation_type, 'direct')

        # define reward type
        # distance, bbox, bbox_distance,
        # self.reward_type = reward_type
        self.reward_type = 'distance'
        self.reward_function = reward.Reward()
        self.trigger_count = 0


        self.count_steps = 0

    def step(self, action):
        obs, rewards, done, info = super(Navigation, self).step(action)

        #detect if the agent collision with environment
        if self.unrealcv.get_hit(self.player[self.protagonist_id]) == 0:
            info['Collision'] = 0
        else:
            info['Collision'] += 1
        # info['Pose'] = self.unrealcv.get_pose(self.cam_id, 'soft') #for virtual camera
        info['Pose'] = self.unrealcv.get_obj_pose(self.player[self.protagonist_id])
        # calculate relative pose
        pose_obs, relative_pose_2d = self.unrealcv.get_pose_states([info['Pose'], self.targets_pos[self.target_list[0]]])
        info['relative_pose'] = np.array([relative_pose_2d[0][1][0], relative_pose_2d[0][1][1],
                                          self.targets_pos[self.target_list[0]][2] - info['Pose'][
                                              2]])  # distance,direction,height : point to player
        # the robot think that it found the target object,the episode is done
        # and get a reward by bounding box size
        # only three times false trigger allowed in every episode
        # if info['Trigger'] > self.trigger_th:
        #     self.trigger_count += 1
        #     # get reward
        #     if 'bbox' in self.reward_type:
        #         object_mask = self.unrealcv.read_image(self.cam_id, 'object_mask')
        #         boxes = self.unrealcv.get_bboxes(object_mask, self.target_list)
        #         info['Reward'], info['Bbox'] = self.reward_function.reward_bbox(boxes)
        #     else:
        #         info['Reward'] = 0
        #
        #     if info['Reward'] > 0 or self.trigger_count > 3:
        #         info['Done'] = True
        #         if info['Reward'] > 0 and self.reset_type == 'waypoint':
        #             self.reset_module.success_waypoint(self.count_steps)
        # else:
        # get reward
        distance, self.target_id = self.select_target_by_distance(info['Pose'][:3], self.targets_pos)
        info['Target'] = self.targets_pos[self.target_id]
        info['Direction'] = misc.get_direction(info['Pose'], self.targets_pos[self.target_id])

        # calculate reward according to the distance to target object
        if 'distance' in self.reward_type:
            # info['Reward'] = self.reward_function.reward_distance(distance)
            relative_oir_norm = np.fabs(info['Direction']) / 90.0
            reward_norm = np.tanh(self.reward_function.reward_distance(distance) - relative_oir_norm)
            info['Reward'] = reward_norm
        else:
            info['Reward'] = 0

        # if collision detected, the episode is done and reward is -1
        if info['Collision'] > COLLISION_THRESHOLD or info['Pose'][2] < self.height/2:
            info['Reward'] = COLLISION_REWARD
            info['Done'] = True

        if distance < SUCCESS_DISTANCE and np.fabs(info['Direction']) < SUCCESS_DIRECTION:
            info['Success'] = True
            info['Done'] = True
            info['Reward'] = SUCCESS_REWARD


        # save the trajectory
        self.trajectory.append(info['Pose'][:6])
        info['Trajectory'] = self.trajectory


        return obs, info['Reward'], info['Done'], info

    def reset(self, ):
        # double check the resetpoint, it is necessary for random reset type
        observations = super(Navigation, self).reset()

        current_pose = self.unrealcv.get_pose(self.cam_id[self.protagonist_id])
        self.targets_pos = self.unrealcv.build_pose_dic(self.target_list)

        self.unrealcv.set_obj_color(self.target_list[0], (255, 255, 255))
        # state = self.unrealcv.get_observation(self.cam_id, self.observation_type)
        observations, self.obj_poses, self.img_show = self.update_observation(self.player_list, self.cam_list, self.cam_flag, self.observation_type)

        self.trajectory = []
        self.trajectory.append(current_pose)
        self.trigger_count = 0
        self.count_steps = 0
        dis2target, self.targetID_last = self.select_target_by_distance(current_pose, self.targets_pos)
        self.reward_function.reset(dis2target)

        return observations

    def seed(self, seed=None):
        return seed

    def render(self, mode='rgb_array', close=False):
        if close:
            self.unreal.close()
        return self.unrealcv.img_color

    def close(self):
        self.unrealcv.client.disconnect()
        self.ue_binary.close()

    def get_action_size(self):
        return len(self.action)

    def select_target_by_distance(self, current_pos, targets_pos):
        # find the nearest target, return distance and targetid
        target_id = list(self.targets_pos.keys())[0]
        # distance_min = self.unrealcv.get_distance(targets_pos[target_id], current_pos, 2)
        distance_min = self.unrealcv.get_distance(targets_pos[target_id], current_pos, 3)

        for key, target_pos in targets_pos.items():
            # distance = self.unrealcv.get_distance(target_pos, current_pos, 2)
            distance = self.unrealcv.get_distance(target_pos, current_pos, 3)
            if distance < distance_min:
                target_id = key
                distance_min = distance
        return distance_min, target_id

