from gym import Wrapper
import numpy as np
import os
import gym_unrealcv
import unrealcv
class RandomPopulationWrapper(Wrapper):
    def __init__(self, env,  num_min=2, num_max=10, random_target=False, random_tracker=False):
        super().__init__(env)
        self.min_num = num_min
        self.max_num = num_max
        self.random_target_id = random_target
        self.random_tracker_id = random_tracker

        self.reset_type = int(env.spec.id.split('-')[-1][-1])
        if 'track_train' in env.unwrapped.env_name and  self.reset_type>0:
            env.unwrapped.objects_list=["cube1", "cube2_7", "cube3", "cube4", "cube5",
                "cylinder1", "cylinder2", "cylinder3", "cylinder4", "cylinder5",
                "cone1", "cone2", "cone3", "cone4", "cone5","sphere1","sphere2","sphere3","sphere4","sphere5"]
            env.unwrapped.env_configs["backgrounds"]=[ "FLOOR","wall1","wall2","wall3","wall4","Cube7_13","Cube8","Cube9","Cube10"]
            env.unwrapped.env_configs["lights"] = ["light1", "light2", "light3", "light4", "light5", "light6"]
            env.unwrapped.textures_list = gym_unrealcv.envs.utils.misc.get_textures(texture_name="textures", docker=env.unwrapped.docker)
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def reset(self, **kwargs):
        env = self.env.unwrapped
        if not env.launched:  # we need to launch the environment
            env.launched = env.launch_ue_env()
            env.init_agents()
            env.init_objects()

        if self.min_num == self.max_num:
            env.num_agents = self.min_num
        else:
            # Randomize the number of agents
            env.num_agents = np.random.randint(self.min_num, self.max_num)
        env.set_population(env.num_agents)
        if self.random_tracker_id:
            env.tracker_id = env.sample_tracker()
        if self.random_target_id:
            new_target = env.sample_target()
            if new_target != env.tracker_id:  # set target object mask to white
                env.unrealcv.build_color_dict(env.player_list)
                env.unrealcv.set_obj_color(env.player_list[env.target_id], env.unrealcv.color_dict[env.player_list[new_target]])
                env.unrealcv.set_obj_color(env.player_list[new_target], [255, 255, 255])
                env.target_id = new_target
        if env.unwrapped.reset_type>0:
            if env.unwrapped.reset_type==1:
                env.unwrapped.environment_augmentation(player_mesh=True, player_texture=True, light=False, background_texture=False,layout=False, layout_texture=False)
            elif env.unwrapped.reset_type==2:
                env.unwrapped.environment_augmentation(player_mesh=True, player_texture=True, light=True,background_texture=False,layout=False, layout_texture=False)
            elif env.unwrapped.reset_type==3:
                env.unwrapped.environment_augmentation(player_mesh=True, player_texture=True, light=True, background_texture=True,layout=False, layout_texture=False)
            elif env.unwrapped.reset_type==4:
                env.unwrapped.environment_augmentation(player_mesh=True, player_texture=True, light=True, background_texture=False,layout=True, layout_texture=True)
            elif env.unwrapped.reset_type == 5:
                env.unwrapped.environment_augmentation(player_mesh=True, player_texture=True, light=True,background_texture=True, layout=True, layout_texture=True)
            elif env.unwrapped.reset_type == 6:
                env.unwrapped.environment_augmentation(player_mesh=True, player_texture=True, light=True,background_texture=False, layout=True, layout_texture=False)

        states = self.env.reset(**kwargs)
        return states