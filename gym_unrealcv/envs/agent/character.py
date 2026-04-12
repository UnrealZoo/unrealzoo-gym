from unrealcv.api import UnrealCv_API
import numpy as np
import math
import time
import json
import re
import logging
from io import BytesIO
import PIL.Image
from gym_unrealcv.envs.utils import misc

logger = logging.getLogger(__name__)


class Character_API(UnrealCv_API):
    def __init__(self, port=9000, ip='127.0.0.1', resolution=(160, 120), comm_mode='tcp'):
        super().__init__(port=port, ip=ip, resolution=resolution, mode=comm_mode)
        self.obstacles = []
        self.targets = []
        self.img_color = np.zeros((resolution[1], resolution[0], 3))
        self.img_depth = np.zeros((resolution[1], resolution[0], 1))
        self.animation_dict = {
            'stand': self.set_standup,
            'jump': self.set_jump,
            'crouch': self.set_crouch,
            'liedown': self.set_liedown,
            'open_door': self.set_open_door,
            'enter_vehicle': self.enter_exit_car,
            'carry':self.carry_body,
            'drop':self.drop_body
        }

    def _request_with_retry(self, cmd, timeout=None, retries=100, delay=0.01):
        for _ in range(retries):
            res = self.client.request(cmd, timeout) if timeout is not None else self.client.request(cmd)
            if res is not None:
                return res
            time.sleep(delay)
        raise RuntimeError(f'UnrealCV request failed after {retries} retries: {cmd}')

    def init_mask_color(self, targets=None):
        if targets == 'all':
            self.targets = self.get_objects()
            self.color_dict = self.build_color_dict(self.targets)
        elif targets is not None:
            self.targets = targets
            self.color_dict = self.build_color_dict(self.targets)

    def get_observation(self, cam_id, observation_type, mode='bmp'):
        """
        Get the observation from the specified camera.

        Args:
            cam_id (int): The ID of the camera.
            observation_type (str): The type of observation to retrieve. Options are 'Color', 'Mask', 'Depth', 'Rgbd', 'Gray', 'Pose'.
            mode (str, optional): The mode in which to retrieve the image. Defaults to 'bmp'.

        Returns:
            np.ndarray: The observation data.
        """
        if observation_type == 'Color':
            self.img_color = state = self.get_image(cam_id, 'lit', mode)
        elif observation_type == 'Mask':
            self.img_color = state = self.get_image(cam_id, 'object_mask', mode)
        elif observation_type == 'Depth':
            self.img_depth = state = self.get_depth(cam_id)
        elif observation_type == 'Rgbd':
            state = self.get_image_multimodal(cam_id, ['lit', 'depth'], [mode, 'npy'])
            self.img_color = state[..., :3]
            self.img_depth = state[..., 3:]
        elif observation_type == 'Gray':
            self.img_color = self.read_image(cam_id, 'lit', mode)
            self.img_gray = self.img_color.mean(2)
            self.img_gray = np.expand_dims(self.img_gray, -1)
        elif observation_type == 'Pose':
            state = self.get_pose() # fake pose
        return state

    def get_pose(self, cam_id, newest=True):
        """
        Get the pose of the specified camera.

        Args:
            cam_id (int): The ID of the camera.
            newest (bool, optional): If True, get the latest camera pose. If False, get the stored camera pose. Defaults to True.

        Returns:
            list: The pose of the camera in the format [x, y, z, roll, yaw, pitch].
        """
        if newest:
            pose = self.get_cam_location(cam_id) + self.get_cam_rotation(cam_id)
        else:
            pose = self.cam[cam_id]['location'] + self.cam[cam_id]['rotation']
        return pose

    # functions for character setting
    def set_max_speed(self, player, speed, return_cmd=False):
        """
        Set the maximum velocity of the agent object.

        Args:
            player (str): The identifier of the player.
            speed (float): The maximum speed to set for the player.
            return_cmd (bool, optional): If True, return the command string instead of executing it. Defaults to False.

        Returns:
            float | str: The speed that was set, or command string when return_cmd is True.
        """
        cmd = f'vbp {player} set_speed {speed}'
        if return_cmd:
            return cmd
        self._request_with_retry(cmd)
        return speed

    def set_acceleration(self, player, acc):
        """
        Set the acceleration of the agent object.

        Args:
            player (str): The identifier of the player.
            acc (float): The acceleration value to set for the player.

        Returns:
            float: The acceleration that was set.
        """
        cmd = f'vbp {player} set_acc {acc}'
        self._request_with_retry(cmd)
        return acc

    def set_appearance(self, player, id):
        """
           Set the appearance of the agent object.

           Args:
               player (str): The identifier of the player.
               id (int): The appearance ID to set for the player.

           Returns:
               int: The appearance ID that was set.
           """
        cmd = f'vbp {player} set_app {id}'
        self._request_with_retry(cmd, -1)
        return id

    def move_cam_2d(self, cam_id, angle, distance):
        """
            Move the camera in 2D as a mobile agent.

            Args:
                cam_id (int): The ID of the camera.
                angle (float): The angle to move the camera.
                distance (float): The distance to move the camera.
        """
        self.move_cam_forward(cam_id, angle, distance, height=0, pitch=0)

    def get_speed(self, player):
        """
            Get the speed of the agent object.

            Args:
                player (str): The identifier of the player.

            Returns:
                float: The speed of the player.
        """
        cmd = f'vbp {player} get_speed'
        res = self._request_with_retry(cmd)
        return self.decoder.string2vector(res)[0]

    def get_angle(self, player):
        """
           Get the angular of the agent object.

           Args:
               player (str): The identifier of the player.

           Returns:
               float: The angle of the player.
        """
        cmd = f'vbp {player} get_angle'
        res = self._request_with_retry(cmd)
        return self.decoder.string2vector(res)[0]

    def reset_player(self, player):
        """
            Reset the agent object.

            Args:
                player (str): The identifier of the player.
        """
        cmd = f'vbp {player} reset'
        self._request_with_retry(cmd)

    def set_phy(self, obj, state):
        """
            Set the physics state of the object.
            Args:
                obj (str): The identifier of the object.
                state (int): The physics state to set (0 or 1).
        """
        cmd = f'vbp {obj} set_phy {state}'
        self._request_with_retry(cmd, -1)

    def set_rotation(self,obj,yaw):
        #set drone's rotation ('set_obj_rotation' doesn't work for drone)
        cmd = f'vbp {obj} set_rotation {yaw}'
        self._request_with_retry(cmd, -1)


    def simulate_physics(self, objects):
        res = [self.set_phy(obj, 1) for obj in objects]

    def set_move_bp(self, player, params, return_cmd=False):
        '''
        new move function, can adapt to different number of params
        2 params: [v_angle, v_linear], used for agents moving in plane, e.g. human, car, animal
        4 params: [v_ x, v_y, v_z, v_yaw], used for agents moving in 3D space, e.g. drone
        '''
        params_str = ' '.join([str(param) for param in params])
        cmd = f'vbp {player} set_move {params_str}'
        if return_cmd:
            return cmd
        self._request_with_retry(cmd, -1)

    # functions for character actions
    def set_jump(self, player, return_cmd=False):
        cmd = f'vbp {player} set_jump'
        if return_cmd:
            return cmd
        self._request_with_retry(cmd, -1)

    def set_crouch(self, player, return_cmd=False):
        cmd = f'vbp {player} set_crouch'
        if return_cmd:
            return cmd
        self._request_with_retry(cmd, -1)

    def set_liedown(self, player, directions=(100, 100), return_cmd=False):
        frontback = directions[0]
        leftright = directions[1]
        cmd = f'vbp {player} set_liedown {frontback} {leftright}'
        if return_cmd:
            return cmd
        self._request_with_retry(cmd, -1)

    def set_standup(self, player, return_cmd=False):
        cmd = f'vbp {player} set_standup'
        if return_cmd:
            return cmd
        self._request_with_retry(cmd, -1)

    def set_animation(self, player, anim_id, return_cmd=False):
        return self.animation_dict[anim_id](player, return_cmd=return_cmd)

    def get_hit(self, player):
        cmd = f'vbp {player} get_hit'
        res = self._request_with_retry(cmd)
        if '1' in res:
            return True
        if '0' in res:
            return False

    def set_random(self, player, value=1):
        cmd = f'vbp {player} set_random {value}'
        self._request_with_retry(cmd, -1)

    def set_interval(self, player, interval):
        cmd = f'vbp {player} set_interval {interval}'
        self._request_with_retry(cmd, -1)

    def init_objects(self, objects):
        self.objects_dict = dict()
        for obj in objects:
            # print (obj)
            self.objects_dict[obj] = self.get_obj_location(obj)
        return self.objects_dict

    def random_obstacles(self, objects, img_dirs, num, area, start_area, texture=False):
        sample_index = np.random.choice(len(objects), num, replace=False)
        for id in sample_index:
            obstacle = objects[id]
            self.obstacles.append(obstacle)
            # texture
            if texture:
                img_dir = img_dirs[np.random.randint(0, len(img_dirs))]
                self.set_texture(obstacle, (1, 1, 1), np.random.uniform(0, 1, 3), img_dir, np.random.randint(1, 4))
            # scale
            # self.set_obj_scale(obstacle, np.random.uniform(0.3, 3, 3))
            self.set_obj_scale(obstacle, np.random.uniform(2, 5, 3))

            # location
            obstacle_loc = [start_area[0], start_area[2], 0]
            while start_area[0] <= obstacle_loc[0] <= start_area[1] and start_area[2] <= obstacle_loc[1] <= start_area[3]:
                obstacle_loc[0] = np.random.uniform(area[0]+200, area[1]-200)
                obstacle_loc[1] = np.random.uniform(area[2]+200, area[3]-200)
                obstacle_loc[2] = np.random.uniform(area[4], area[5]) -150
            self.set_obj_location(obstacle, obstacle_loc)
            time.sleep(0.1)

    def clean_obstacles(self):
        for obj in self.obstacles:
            self.set_obj_location(obj, self.objects_dict[obj])
        self.obstacles = []

    def new_obj(self, obj_class_name, obj_name, loc, rot=[0, 0, 0]):
        # spawn, set obj pose, enable physics
        [x, y, z] = loc
        [pitch, yaw, roll] = rot
        if obj_class_name =="bp_character_C" or obj_class_name =="target_C":
            cmd = [f'vset /objects/spawn {obj_class_name} {obj_name}',
                   f'vset /object/{obj_name}/location {x} {y} {z}',
                   f'vset /object/{obj_name}/rotation {pitch} {yaw} {roll}',
                   f'vbp {obj_name} set_phy 0'
                   ]
        else:
            cmd = [f'vset /objects/spawn {obj_class_name} {obj_name}',
                   f'vset /object/{obj_name}/location {x} {y} {z}',
                   f'vset /object/{obj_name}/rotation {pitch} {yaw} {roll}',
                   f'vbp {obj_name} set_phy 1'
                   ]
        self.client.request(cmd, -1)
        return obj_name

    def set_cam(self, obj, loc=[0, 30, 70], rot=[0, 0, 0], return_cmd=False):
        # set the camera pose relative to a actor
        x, y, z = loc
        roll, pitch, yaw = rot
        cmd = f'vbp {obj} set_cam {x} {y} {z} {roll} {pitch} {yaw}'
        if return_cmd:
            return cmd
        res = self.client.request(cmd, -1)
        return res

    def adjust_fov(self, cam_id, delta_fov, min_max=[45, 135]):  # increase/decrease fov
        return self.set_fov(cam_id, np.clip(self.cam[cam_id]['fov']+delta_fov, min_max[0], min_max[1]))

    def stop_car(self, obj):
        cmd = f'vbp {obj} set_stop'
        res = self.client.request(cmd, -1)
        return res

    def nav_to_goal(self, obj, loc): # navigate the agent to a goal location
        # Assign the agent a navigation goal, and use Navmesh to automatically control its movement to reach the goal via the shortest path.
        # The goal should be reachable in the environment.
        x, y, z = loc
        cmd = f'vbp {obj} nav_to_goal {x} {y} {z}'
        res = self.client.request(cmd, -1)
        return res
    def nav_to_goal_bypath(self, obj, loc): # navigate the agent to a goal location
        # Assign the agent a navigation goal, and use Navmesh to automatically control its movement to reach the goal via the shortest path.
        # The goal should be reachable in the environment.
        x, y, z = loc
        cmd = f'vbp {obj} nav_to_goal_bypath {x} {y} {z}'
        res = self.client.request(cmd, -1)
        return res
    def nav_to_random(self, obj, radius, loop): # navigate the agent to a random location
        # Agent randomly selects a point within its own radius range for navigation.
        # The loop parameter controls whether continuous navigation is performed.（True for continuous navigation).
        # Return with the randomly sampled location.
        cmd = f'vbp {obj} nav_random {radius} {loop}'
        res = self.client.request(cmd)
        return res
    def nav_to_obj(self, obj, target_obj, distance=200): # navigate the agent to a target object
        # Assign the agent a navigation goal, and use Navmesh to automatically control its movement to reach the goal via the shortest path.
        cmd = f'vbp {obj} nav_to_target {target_obj} {distance}'
        res = self.client.request(cmd, -1)
        return res

    def nav_random(self, player, radius, loop): # navigate the agent to a random location
        # Agent randomly selects a point within its own radius range for navigation.
        # The loop parameter controls whether continuous navigation is performed.（True for continuous navigation).
        # Return with the randomly sampled location.
        cmd = f'vbp {player} nav_random {radius} {loop}'
        res = self.client.request(cmd)
        return self.decoder.string2vector(res)

    def generate_nav_goal(self, player, radius_max,radius_min):  # navigate the agent to a random location
        # Agent randomly selects a point within its own radius range for navigation.
        # The loop parameter controls whether continuous navigation is performed.（True for continuous navigation).
        # Return with the randomly sampled location.
        cmd = f'vbp {player} generate_nav_goal {radius_max} {radius_min} '
        res = self.client.request(cmd)
        answer_dict = json.loads(res)
        loc = answer_dict.get("nav_goal") or answer_dict.get("Nav_goal")
        if loc is None:
            raise KeyError(f"Missing nav goal key in response, available keys: {list(answer_dict.keys())}")
        coordinates = re.findall(r"[-+]?\d*\.\d+|\d+", loc)
        # Convert the numbers to floats and store them in an array
        coordinates = [float(coordinate) for coordinate in coordinates]
        return coordinates[0],coordinates[1],coordinates[2]

    def set_max_nav_speed(self, obj, max_vel): # set the maximum navigation speed of the car
        cmd = f'vbp {obj} set_nav_speed {max_vel}'
        res = self.client.request(cmd, -1)
        return res

    def enter_exit_car(self, obj, player_index):
        # enter or exit the car for a player.
        # If the player is already in the car, it will exit the car. Otherwise, it will enter the car.
        cmd = f'vbp {obj} enter_exit_car {player_index}'
        res = self.client.request(cmd, -1)
        return res

    def set_open_door(self, player, state, return_cmd=False):
        # state: 0 close, 1 open
        cmd = f'vbp {player} set_open_door {state}'
        if return_cmd:
            return cmd
        else:
            self.client.request(cmd, -1)
    def carry_body(self,player,return_cmd=False):
        cmd = f'vbp {player} carry_body'
        if return_cmd:
            return cmd
        else:
            self.client.request(cmd, -1)

    def drop_body(self,player,return_cmd=False):
        cmd = f'vbp {player} drop_body'
        if return_cmd:
            return cmd
        else:
            self.client.request(cmd, -1)
    def Is_picked(self,player,return_cmd = False):
        cmd = f'vbp {player} is_picked'
        if return_cmd:
            return cmd
        else:
            res = self.client.request(cmd)
            if '1' in res:
                return True
            if '0' in res:
                return False
    def is_carrying(self,player,return_cmd = False):
        cmd = f'vbp {player} is_carrying'
        if return_cmd:
            return cmd
        else:
            res = self.client.request(cmd)
            if '1' in res:
                return True
            if '0' in res:
                return False

    def set_viewport(self, player):
        # set the game window to the player's view
        cmd = f'vbp {player} set_viewport'
        res = self.client.request(cmd, -1)
        return res

    def get_pose_img_batch(self, objs_list, cam_ids, img_flag=[False, True, False, False]):
        # get pose and image of objects in objs_list from cameras in cam_ids
        cmd_list = []
        decoder_list = []
        [use_cam_pose, use_color, use_mask, use_depth] = img_flag
        for obj in objs_list:
            cmd_list.extend([self.get_obj_location(obj, True),
                             self.get_obj_rotation(obj, True)])

        for cam_id in cam_ids:
            if cam_id < 0:
                continue
            if use_cam_pose:
                cmd_list.extend([self.get_cam_location(cam_id, return_cmd=True),
                                 self.get_cam_rotation(cam_id, return_cmd=True)])
            if use_color:
                cmd_list.append(self.get_image(cam_id, 'lit', 'bmp', return_cmd=True))
            if use_mask:
                cmd_list.append(self.get_image(cam_id, 'object_mask', 'png', return_cmd=True))
            if use_depth:
                cmd_list.append(f'vget /camera/{cam_id}/depth npy')
                # cmd_list.append(self.get_image(cam_id, 'depth', 'bmp', return_cmd=True))

        decoders = [self.decoder.decode_map[self.decoder.cmd2key(cmd)] for cmd in cmd_list]
        try:
            res_list = self.batch_cmd(cmd_list, decoders)
        except Exception as exc:
            logger.error('batch cmd error: %s', exc)
            raise
        obj_pose_list = []
        cam_pose_list = []
        img_list = []
        mask_list = []
        depth_list = []
        # start to read results
        start_point = 0
        for i, obj in enumerate(objs_list):
            obj_pose_list.append(res_list[start_point] + res_list[start_point+1])
            start_point += 2
        for i, cam_id in enumerate(cam_ids):
            # print(cam_id)
            if cam_id < 0:
                img_list.append(np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8))
                continue
            if use_cam_pose:
                cam_pose_list.append(res_list[start_point] + res_list[start_point+1])
                start_point += 2
            if use_color:
                # image = self.decoder.decode_bmp(res_list[start_point])
                img_list.append(res_list[start_point])
                start_point += 1
            if use_mask:
                # image = self.decoder.decode_bmp(res_list[start_point])
                mask_list.append(res_list[start_point])
                start_point += 1
            if use_depth:
                # image = 1 / self.decoder.decode_depth(res_list[start_point],bytesio=False)
                # image = self.decoder.decode_depth(res_list[start_point],bytesio=False)
                image = res_list[start_point]
                if image.ndim == 2:
                    image = np.expand_dims(image, axis=-1)
                depth_list.append(image)  # 500 is the default max depth of most depth cameras
                # depth_list.append(res_list[start_point])  # 500 is the default max depth of most depth cameras
                start_point += 1

        return obj_pose_list, cam_pose_list, img_list, mask_list, depth_list

    # Domain Randomization Functions: randomize texture
    def set_texture(self, player, color=(1, 1, 1), param=(0, 0, 0), picpath=None, tiling=1, e_num=0): #[r, g, b, meta, spec, rough, tiling, picpath]
        param = param / param.max()
        r, g, b = color
        meta, spec, rough = param
        cmd = f'vbp {player} set_mat {e_num} {r} {g} {b} {meta} {spec} {rough} {tiling} {picpath}'
        self.client.request(cmd, -1)

    def set_light(self, obj, direction, intensity, color): # param num out of range
        [roll, yaw, pitch] = direction
        color = color / color.max()
        [r, g, b] = color
        cmd = f'vbp {obj} set_light {roll} {yaw} {pitch} {intensity} {r} {g} {b}'
        self.client.request(cmd, -1)

    def random_texture(self, backgrounds, img_dirs, num=5):
        if num < 0:
            sample_index = range(len(backgrounds))
        else:
            sample_index = np.random.choice(len(backgrounds), num, replace=False)
        for id in sample_index:
            obj = backgrounds[id]
            img_dir = img_dirs[np.random.randint(0, len(img_dirs))]
            self.set_texture(obj, (1, 1, 1), np.random.uniform(0, 1, 3), img_dir, np.random.randint(1, 4))
            time.sleep(0.03)

    def random_player_texture(self, player, img_dirs, num):
        sample_index = np.random.choice(5, num)
        for id in sample_index:
            img_dir = img_dirs[np.random.randint(0, len(img_dirs))]
            self.set_texture(player, (1, 1, 1), np.random.uniform(0, 1, 3),
                             img_dir, np.random.randint(2, 6), id)
            time.sleep(0.03)

    def random_character(self, player):  # appearance, speed, acceleration
        self.set_max_speed(player, np.random.randint(40, 100))
        self.set_acceleration(player, np.random.randint(100, 300))

    def random_lit(self, light_list):
        for lit in light_list:
            if 'sky' in lit:
                self.set_skylight(lit, [1, 1, 1], np.random.uniform(1, 10))
            else:
                lit_direction = np.random.uniform(-1, 1, 3)
                if 'directional' in lit:
                    lit_direction[0] = lit_direction[0] * 60
                    lit_direction[1] = lit_direction[1] * 80
                    lit_direction[2] = lit_direction[2] * 60
                else:
                    lit_direction *= 180
                self.set_light(lit, lit_direction, np.random.uniform(1, 5), np.random.uniform(0.3, 1, 3))

    def set_skylight(self, obj, color, intensity): # param num out of range
        [r, g, b] = color
        cmd = f'vbp {obj} set_light {r} {g} {b} {intensity}'
        self.client.request(cmd, -1)

    def get_obj_speed(self,obj):
        cmd = f'vbp {obj} get_speed'
        res = self.client.request(cmd)
        answer_dict = json.loads(res)
        speed = float(answer_dict["Speed"])

        return speed
    def check_visibility(self, tracker_cam_id,target_obj):
        mask = self.read_image(tracker_cam_id, 'object_mask', 'fast')
        mask, bbox = self.get_bbox(mask, target_obj, normalize=False)
        mask_percent = mask.sum()/(self.resolution[0] * self.resolution[1])
        return mask_percent
    def read_image(self, cam_id, viewmode, mode='direct'):
            # cam_id:0 1 2 ...
            # viewmode:lit,  normal, depth, object_mask
            # mode: direct, file
            res = None
            if mode == 'direct': # get image from unrealcv in png format
                cmd = f'vget /camera/{cam_id}/{viewmode} png'
                image = self.decode_png(self.client.request(cmd))

            elif mode == 'file': # save image to file and read it
                cmd = f'vget /camera/{cam_id}/{viewmode} {viewmode}{self.ip}.png'
                if self.docker:
                    img_dirs_docker = self.client.request(cmd)
                    img_dirs = self.envdir + img_dirs_docker[7:]
                else :
                    img_dirs = self.client.request(cmd)
                image = cv2.imread(img_dirs)
            elif mode == 'fast': # get image from unrealcv in bmp format
                cmd = f'vget /camera/{cam_id}/{viewmode} bmp'
                image = self.decode_bmp(self.client.request(cmd))
            return image

    def decode_png(self, res):  # decode png image
        img = np.asarray(PIL.Image.open(BytesIO(res)))
        img = img[:, :, :-1]  # delete alpha channel
        img = img[:, :, ::-1]  # transpose channel order
        return img

    def decode_bmp(self, res, channel=4):  # decode bmp image
        img = np.frombuffer(res, dtype=np.uint8)
        img = img[-self.resolution[1] * self.resolution[0] * channel:]
        img = img.reshape(self.resolution[1], self.resolution[0], channel)
        return img[:, :, :-1]  # delete alpha channel

    def decode_depth(self, res):  # decode depth image
        depth = np.frombuffer(res, np.float32)
        depth = depth[-self.resolution[1] * self.resolution[0]:]
        depth = depth.reshape(self.resolution[1], self.resolution[0], 1)
        return depth
    def set_location(self, cam_id, loc):  # set camera location, loc=[x,y,z]
        [x, y, z] = loc
        cmd = f'vset /camera/{cam_id}/location {x} {y} {z}'
        self.client.request(cmd, -1)
        self.cam[cam_id]['location'] = loc
    def get_pose_states(self, obj_pos):
        # get the relative pose of each agent and the absolute location and orientation of the agent
        pose_obs = []
        player_num = len(obj_pos)
        np.zeros((player_num, player_num, 2))
        relative_pose = np.zeros((player_num, player_num, 2))
        for j in range(player_num):
            vectors = []
            for i in range(player_num):
                obs, distance, direction = self.get_relative(obj_pos[j], obj_pos[i])
                yaw = obj_pos[j][4]/180*np.pi
                # rescale the absolute location and orientation
                abs_loc = [obj_pos[i][0], obj_pos[i][1],
                           obj_pos[i][2], np.cos(yaw), np.sin(yaw)]
                obs = obs + abs_loc
                vectors.append(obs)
                relative_pose[j, i] = np.array([distance, direction])
            pose_obs.append(vectors)

        return np.array(pose_obs), relative_pose
    def get_relative(self, pose0, pose1):  # pose0-centric
        """
        Get the relative pose between two objects, pose0 is the reference object.

        Args:
            pose0 (list): Pose of the reference object (the center of the coordinate system).
            pose1 (list): Pose of the target object.

        Returns:
            tuple: Relative observation vector, distance, and angle.
        """
        delt_yaw = pose1[4] - pose0[4]
        angle = misc.get_direction(pose0, pose1)
        distance = self.get_distance(pose1, pose0, 3)
        obs_vector = [np.sin(delt_yaw/180*np.pi), np.cos(delt_yaw/180*np.pi),
                      np.sin(angle/180*np.pi), np.cos(angle/180*np.pi),
                      distance]
        return obs_vector, distance, angle