__version__ = "2.0.3"
from gym_unrealcv._gym_compat import gym, register
import logging
import os
import re

logger = logging.getLogger(__name__)
use_docker = False  # True: use nvidia docker   False: do not use nvidia-docker

# ---------------------------------------------------------------------------
# Map catalogue – single source of truth for all available Unreal maps.
# ---------------------------------------------------------------------------
MAPS = [
    'Greek_Island', 'supermarket', 'Brass_Gardens', 'Brass_Palace', 'Brass_Streets',
    'EF_Gus', 'EF_Lewis_1', 'EF_Lewis_2', 'EF_Grounds', 'TemplePlaza', 'Eastern_Garden',
    'Western_Garden', 'Colosseum_Desert', 'Desert_ruins', 'SchoolGymDay', 'Venice',
    'VictorianTrainStation', 'Stadium', 'IndustrialArea', 'ModularBuilding', 'DowntownWest',
    'TerrainDemo', 'InteriorDemo_NEW', 'AncientRuins', 'Grass_Hills', 'ChineseWaterTown_Ver1',
    'ContainerYard_Night', 'ContainerYard_Day', 'Old_Factory_01', 'racing_track', 'Watermills',
    'WildWest', 'SunsetMap', 'Hospital', 'Medieval_Castle', 'Real_Landscape',
    'UndergroundParking', 'Demonstration_Castle', 'Demonstration_Cave', 'PlatFormHangar',
    'PlatformFactory', 'demonstration_BUNKER', 'Arctic', 'Medieval_Daytime',
    'Medieval_Nighttime', 'ModularGothic_Day', 'ModularGothic_Night', 'UltimateFarming',
    'RuralAustralia_Example_01', 'RuralAustralia_Example_02', 'RuralAustralia_Example_03',
    'LV_Soul_Cave', 'Dungeon_Demo_00', 'SwimmingPool', 'DesertMap', 'RainMap', 'SnowMap',
    'ModularVictorianCity', 'SuburbNeighborhood_Day', 'SuburbNeighborhood_Night',
    'Storagehouse', 'ModularNeighborhood', 'ModularSciFiVillage', 'ModularSciFiSeason1',
    'LowPolyMedievalInterior_1', 'QA_Holding_Cells_A', 'ParkingLot', 'Demo_Roof',
    'MiddleEast', 'Lighthouse', 'Cabin_Lake', 'UniversityClassroom', 'Tokyo',
    'CommandCenter', 'JapanTrainStation_Optimised', 'Hotel_Corridor', 'Museum',
    'ForestGasStation', 'KoreanPalace', 'CourtYard', 'Chinese_Landscape_Demo',
    'EnglishCollege', 'OperaHouse', 'AsianTemple', 'Pyramid', 'PlanetOutDoor',
    'Map_ChemicalPlant_1', 'Hangar', 'Science_Fiction_valley_town',
    'RussianWinterTownDemo01', 'LookoutTower', 'LV_Bazaar', 'OperatingRoom',
    'PostSoviet_Village', 'Old_Town', 'AsianMedivalCity', 'StonePineForest',
    'TemplesOfCambodia_01_01_Exterior', 'AbandonedDistrict', 'FlexibleRoom',
]

TASKS = ['Rendezvous', 'Rescue', 'Track', 'Navigation', 'NavigationMulti']
OBSERVATIONS = ['Color', 'Depth', 'Rgbd', 'Gray', 'CG', 'Mask', 'Pose', 'MaskDepth', 'ColorMask']
ACTIONS = ['Discrete', 'Continuous', 'Mixed']
_MAPS_SET = frozenset(MAPS)

# ---------------------------------------------------------------------------
# Lazy registration helpers
# ---------------------------------------------------------------------------
# Instead of eagerly registering ~116k env specs at import time (which took
# ~20 s), we *only* register an env when it is first requested via gym.make().
# We do this by monkey-patching the gym registry's __contains__ / spec lookup
# so that it can recognise our naming convention and register on-the-fly.


def _parse_and_register(env_id: str) -> bool:
    """Attempt to parse *env_id* against known naming conventions and register it on-the-fly.

    Returns True if the env was successfully registered, False otherwise.
    """
    # ---- Legacy Robot Arm envs ----
    m = re.fullmatch(r'UnrealArm-(?P<action>Discrete|Continuous)(?P<obs>Pose|Color|Depth|Rgbd)-v(?P<ver>[0-2])', env_id)
    if m:
        register(
            id=env_id,
            entry_point='gym_unrealcv.envs:UnrealCvRobotArm_reach',
            kwargs={
                'setting_file': os.path.join('robotarm', 'robotarm_reach.json'),
                'action_type': m.group('action'),
                'observation_type': m.group('obs'),
                'docker': use_docker,
                'version': int(m.group('ver')),
            },
            max_episode_steps=100,
        )
        return True

    # ---- Legacy spline tracking (ICML 2018) ----
    m = re.fullmatch(
        r'UnrealTrack-(?P<env>City[12])(?P<target>Malcom|Stefani)(?P<path>Path[12])-'
        r'(?P<action>Discrete|Continuous)(?P<obs>Color|Depth|Rgbd)-v(?P<reset>[01])',
        env_id,
    )
    if m:
        reset_name = ['Static', 'Random'][int(m.group('reset'))]
        register(
            id=env_id,
            entry_point='gym_unrealcv.envs:UnrealCvTracking_spline',
            kwargs={
                'setting_file': os.path.join('tracking', 'v0',
                                             f"{m.group('env')}{m.group('target')}{m.group('path')}.json"),
                'reset_type': reset_name,
                'action_type': m.group('action'),
                'observation_type': m.group('obs'),
                'reward_type': 'distance',
                'docker': use_docker,
            },
            max_episode_steps=3000,
        )
        return True

    # ---- Multi-camera tracking (AAAI 2020) ----
    _mc_navs = {'Random', 'Goal', 'Internal', 'None',
                'RandomInterval', 'GoalInterval', 'InternalInterval', 'NoneInterval'}
    m = re.fullmatch(
        r'Unreal(?P<env>MCRoom|Garden|UrbanTree)-'
        r'(?P<action>Discrete|Continuous)(?P<obs>Color|Depth|Rgbd|Gray)(?P<nav>\w+)-v(?P<reset>[0-6])',
        env_id,
    )
    if m and m.group('nav') in _mc_navs:
        register(
            id=env_id,
            entry_point='gym_unrealcv.envs:UnrealCvMC',
            kwargs={
                'setting_file': os.path.join('tracking', 'multicam', f"{m.group('env')}.json"),
                'reset_type': int(m.group('reset')),
                'action_type': m.group('action'),
                'observation_type': m.group('obs'),
                'reward_type': 'distance',
                'docker': use_docker,
                'nav': m.group('nav'),
            },
            max_episode_steps=500,
        )
        return True

    # ---- MCMT tracking ----
    _mcmt_navs = _mc_navs - {'NoneInterval'}
    m = re.fullmatch(
        r'UnrealMC(?P<env>FlexibleRoom|Garden|UrbanTree)-'
        r'(?P<action>Discrete|Continuous)(?P<obs>Color|Depth|Rgbd|Gray)(?P<nav>\w+)-v(?P<reset>[0-6])',
        env_id,
    )
    if m and m.group('nav') in _mcmt_navs:
        register(
            id=env_id,
            entry_point='gym_unrealcv.envs:UnrealCvMultiCam',
            kwargs={
                'setting_file': os.path.join('tracking', 'mcmt', f"{m.group('env')}.json"),
                'reset_type': int(m.group('reset')),
                'action_type': m.group('action'),
                'observation_type': m.group('obs'),
                'reward_type': 'distance',
                'docker': use_docker,
                'nav': m.group('nav'),
            },
            max_episode_steps=500,
        )
        return True

    # ---- Generic agent env (UnrealAgent-<map>-...) ----
    _obs_pat = '|'.join(OBSERVATIONS)
    _act_pat = '|'.join(ACTIONS)
    m = re.fullmatch(
        rf'UnrealAgent-(?P<env>[A-Za-z0-9_]+)-(?P<action>{_act_pat})(?P<obs>{_obs_pat})-v(?P<reset>[0-6])',
        env_id,
    )
    if m and m.group('env') in _MAPS_SET:
        register(
            id=env_id,
            entry_point='gym_unrealcv.envs:UnrealCv_base',
            kwargs={
                'setting_file': os.path.join('env_config', f"{m.group('env')}.json"),
                'action_type': m.group('action'),
                'observation_type': m.group('obs'),
                'reset_type': int(m.group('reset')),
            },
            max_episode_steps=500,
        )
        return True

    # ---- Task-oriented envs (Unreal<Task>-<map>-...) ----
    _task_pat = '|'.join(TASKS)
    m = re.fullmatch(
        rf'Unreal(?P<task>{_task_pat})-(?P<env>[A-Za-z0-9_]+)-(?P<action>{_act_pat})(?P<obs>{_obs_pat})-v(?P<reset>[0-6])',
        env_id,
    )
    if m and m.group('env') in _MAPS_SET:
        task = m.group('task')
        max_steps = 1000 if task == 'Navigation' else 500
        register(
            id=env_id,
            entry_point=f'gym_unrealcv.envs:{task}',
            kwargs={
                'env_file': os.path.join(task, f"{m.group('env')}.json"),
                'action_type': m.group('action'),
                'observation_type': m.group('obs'),
                'reset_type': int(m.group('reset')),
            },
            max_episode_steps=max_steps,
        )
        return True

    return False


# Monkey-patch the public `gym.spec` API for lazy on-demand registration.
# This is more stable across gym versions than patching private registry internals.
_original_spec = gym.spec


def _lazy_spec(env_id: str):
    """Look up *env_id*; if missing, try to parse/register and retry once."""
    try:
        return _original_spec(env_id)
    except Exception:
        if _parse_and_register(env_id):
            logger.debug('Lazily registered %s', env_id)
            return _original_spec(env_id)
        raise


gym.spec = _lazy_spec
