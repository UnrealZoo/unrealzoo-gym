__version__ = "2.0.3"
from gym.envs.registration import register
import logging
import os
from gym_unrealcv.envs.utils.misc import load_env_setting
logger = logging.getLogger(__name__)
use_docker = False  # True: use nvidia docker   False: do not use nvidia-docker



# ------------------------------------------------------------------
# Robot Arm
# "CRAVES: Controlling Robotic Arm With a Vision-Based Economic System", CVPR 2019
for action in ['Discrete', 'Continuous']:  # action type
    for obs in ['Pose', 'Color', 'Depth', 'Rgbd']:
        for i in range(3):
            register(
                    id='UnrealArm-{action}{obs}-v{version}'.format(action=action, obs=obs, version=i),
                    entry_point='gym_unrealcv.envs:UnrealCvRobotArm_reach',
                    kwargs={'setting_file': os.path.join('robotarm', 'robotarm_reach.json'),
                            'action_type': action,
                            'observation_type': obs,
                            'docker': use_docker,
                            'version': i
                            },
                    max_episode_steps=100
                        )

# -----------------------------------------------------------------------
# Tracking
# "End-to-end Active Object Tracking via Reinforcement Learning", ICML 2018
for env in ['City1', 'City2']:
    for target in ['Malcom', 'Stefani']:
        for action in ['Discrete', 'Continuous']:  # action type
            for obs in ['Color', 'Depth', 'Rgbd']:  # observation type
                for path in ['Path1', 'Path2']:  # observation type
                    for i, reset in enumerate(['Static', 'Random']):
                        register(
                            id='UnrealTrack-{env}{target}{path}-'
                               '{action}{obs}-v{reset}'.format(env=env, target=target, path=path,
                                                               action=action, obs=obs, reset=i),
                            entry_point='gym_unrealcv.envs:UnrealCvTracking_spline',
                            kwargs={'setting_file': os.path.join('tracking', 'v0', f'{env}{target}{path}.json'),
                                    'reset_type': reset,
                                    'action_type': action,
                                    'observation_type': obs,
                                    'reward_type': 'distance',
                                    'docker': use_docker,
                                    },
                            max_episode_steps=3000
                            )


# "Pose-Assisted Multi-Camera Collaboration for Active Object Tracking", AAAI 2020
for env in ['MCRoom', 'Garden', 'UrbanTree']:
    for i in range(7):  # reset type
        for action in ['Discrete', 'Continuous']:  # action type
            for obs in ['Color', 'Depth', 'Rgbd', 'Gray']:  # observation type
                for nav in ['Random', 'Goal', 'Internal', 'None',
                            'RandomInterval', 'GoalInterval', 'InternalInterval', 'NoneInterval']:
                    name = 'Unreal{env}-{action}{obs}{nav}-v{reset}'.format(env=env, action=action, obs=obs, nav=nav, reset=i)
                    setting_file = os.path.join('tracking', 'multicam', f'{env}.json')
                    register(
                        id=name,
                        entry_point='gym_unrealcv.envs:UnrealCvMC',
                        kwargs={'setting_file': setting_file,
                                'reset_type': i,
                                'action_type': action,
                                'observation_type': obs,
                                'reward_type': 'distance',
                                'docker': use_docker,
                                'nav': nav
                                },
                        max_episode_steps=500
                    )

for env in ['FlexibleRoom', 'Garden', 'UrbanTree']:
    for i in range(7):  # reset type
        for action in ['Discrete', 'Continuous']:  # action type
            for obs in ['Color', 'Depth', 'Rgbd', 'Gray']:  # observation type
                for nav in ['Random', 'Goal', 'Internal', 'None',
                            'RandomInterval', 'GoalInterval', 'InternalInterval']:
                    name = 'UnrealMC{env}-{action}{obs}{nav}-v{reset}'.format(env=env, action=action, obs=obs, nav=nav, reset=i)
                    setting_file = os.path.join('tracking', 'mcmt', f'{env}.json')
                    register(
                        id=name,
                        entry_point='gym_unrealcv.envs:UnrealCvMultiCam',
                        kwargs={'setting_file': setting_file,
                                'reset_type': i,
                                'action_type': action,
                                'observation_type': obs,
                                'reward_type': 'distance',
                                'docker': use_docker,
                                'nav': nav
                                },
                        max_episode_steps=500
                    )

maps = [
    'Greek_Island', 'supermarket', 'Brass_Gardens', 'Brass_Palace', 'Brass_Streets',
            'EF_Gus', 'EF_Lewis_1', 'EF_Lewis_2', 'EF_Grounds', 'TemplePlaza','Eastern_Garden', 'Western_Garden', 'Colosseum_Desert',
            'Desert_ruins', 'SchoolGymDay', 'Venice', 'VictorianTrainStation', 'Stadium', 'IndustrialArea', 'ModularBuilding',
             'DowntownWest', 'TerrainDemo', 'InteriorDemo_NEW', 'AncientRuins', 'Grass_Hills', 'ChineseWaterTown_Ver1',
            'ContainerYard_Night', 'ContainerYard_Day', 'Old_Factory_01', 'racing_track', 'Watermills', 'WildWest',
            'SunsetMap', 'Hospital', 'Medieval_Castle', 'Real_Landscape', 'UndergroundParking', 'Demonstration_Castle',
            'Demonstration_Cave','PlatFormHangar', 'PlatformFactory','demonstration_BUNKER','Arctic', 'Medieval_Daytime',
            'Medieval_Nighttime', 'ModularGothic_Day', 'ModularGothic_Night',
            'UltimateFarming', 'RuralAustralia_Example_01', 'RuralAustralia_Example_02', 'RuralAustralia_Example_03',
            'LV_Soul_Cave', 'Dungeon_Demo_00', 'SwimmingPool', 'DesertMap', 'RainMap', 'SnowMap', 'ModularVictorianCity',
            'SuburbNeighborhood_Day', 'SuburbNeighborhood_Night', 'Storagehouse','ModularNeighborhood',
            'ModularSciFiVillage','ModularSciFiSeason1',  'LowPolyMedievalInterior_1','QA_Holding_Cells_A', 'ParkingLot','Demo_Roof','MiddleEast','Lighthouse',
            'Cabin_Lake','UniversityClassroom','Tokyo','CommandCenter','JapanTrainStation_Optimised','Hotel_Corridor','Museum','ForestGasStation',
            'KoreanPalace','CourtYard','Chinese_Landscape_Demo','EnglishCollege','OperaHouse','AsianTemple','Pyramid','PlanetOutDoor',
            'Map_ChemicalPlant_1','Hangar','Science_Fiction_valley_town','RussianWinterTownDemo01','LookoutTower','LV_Bazaar','OperatingRoom',
            'PostSoviet_Village','Old_Town','AsianMedivalCity','StonePineForest','TemplesOfCambodia_01_01_Exterior','AbandonedDistrict','FlexibleRoom'
        ]
Tasks = ['Rendezvous', 'Rescue', 'Track','Navigation','NavigationMulti']
Observations = ['Color', 'Depth', 'Rgbd', 'Gray', 'CG', 'Mask', 'Pose','MaskDepth','ColorMask']
Actions = ['Discrete', 'Continuous', 'Mixed']
# Env for general purpose active object tracking
# Base env for general purpose multi-agent interaction
for env in maps:
    for i in range(7):  # reset type
        for action in Actions:  # action type
            for obs in Observations:  # observation type
                        name = 'UnrealAgent-{env}-{action}{obs}-v{reset}'.format(env=env, action=action, obs=obs, target=target, reset=i)
                        setting_file = os.path.join('env_config', f'{env}.json')
                        register(
                            id=name,
                            entry_point='gym_unrealcv.envs:UnrealCv_base',
                            kwargs={'setting_file': setting_file,
                                    'action_type': action,
                                    'observation_type': obs,
                                    'reset_type': i,
                                    },
                            max_episode_steps=500
                            )
# Task-oriented envs
for env in maps:
    for i in range(7):  # reset type
        for action in Actions:  # action type
            for obs in Observations:  # observation type
                for task in Tasks:
                        name = f'Unreal{task}-{env}-{action}{obs}-v{i}'
                        setting_file = os.path.join(task, f'{env}.json')
                        if task =='Navigation':
                            register(
                                id=name,
                                entry_point=f'gym_unrealcv.envs:{task}',
                                kwargs={'env_file': setting_file,
                                        'action_type': action,
                                        'observation_type': obs,
                                        'reset_type': i,
                                        },
                                max_episode_steps=1000
                            )
                        else:
                            register(
                                id=name,
                                entry_point=f'gym_unrealcv.envs:{task}',
                                kwargs={'env_file': setting_file,
                                        'action_type': action,
                                        'observation_type': obs,
                                        'reset_type': i,
                                        },
                                max_episode_steps=500
                                )
