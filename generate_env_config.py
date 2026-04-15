import re
import time
from unrealcv.api import UnrealCv_API
from unrealcv.launcher import RunUnreal
from unrealcv.util import parse_resolution
import argparse
import json
import copy
import numpy as np
import os
import sys
os.environ['UnrealEnv']='E:\\UnrealEnv'
'''
Probe a map (navmesh, doors, cameras) and write env JSON. Agents are template-only: no name/cam_id;
class_name and asset_path come from dicts below (no pre-placed pawns required).
'''

# --- Binary paths (relative to UnrealEnv), per platform ---
# Edit these when packaging names or folder layout change. Used for JSON fields and for default launch binary.
# Use the literal substring {map} anywhere you need the current --env-map name inserted.
ENV_BIN_TEMPLATE_LINUX = "UnrealZoo_UE5_6_Linux_v3.0.0/Linux/UnrealZoo_UE5_6/Binaries/Linux/UnrealZoo_UE5_6"
ENV_BIN_TEMPLATE_WIN = "UnrealZoo_UE5_6_Win64_v3.0.0/UnrealZoo_UE5_6/Binaries/Win64/UnrealZoo_UE5_6.exe"
ENV_BIN_TEMPLATE_MAC = "/wait/for/update"


def format_env_bin_template(template: str, env_map: str) -> str:
    """Apply {map} substitution for paths written to env JSON."""
    return template.replace("{map}", env_map)


def env_bin_template_for_platform() -> str:
    """Template used for this OS when resolving the launch binary under UnrealEnv."""
    if sys.platform == "win32":
        return ENV_BIN_TEMPLATE_WIN
    if sys.platform == "darwin":
        return ENV_BIN_TEMPLATE_MAC
    return ENV_BIN_TEMPLATE_LINUX


def launch_binary_abs(unreal_env: str, template: str, env_map: str) -> str:
    """UnrealEnv + relative template path (with {map} applied), normalized for the current OS."""
    rel = format_env_bin_template(template, env_map)
    rel_norm = rel.replace("\\", "/")
    parts = [p for p in rel_norm.split("/") if p]
    return os.path.normpath(os.path.join(unreal_env, *parts))


# --- spawn_from_path: use asset_path dict (strings or lists); ** = prefix placeholder, optional --asset-prefix ---
# "animal" shares BP_Character with "player"; UE differentiates meshes via set_animal_app vs set_app (see Character_API / base_env).
class_name = {
    "player": "bp_character_C",
    "animal": "bp_character_C",
    "drone": "BP_drone01_C",
    # "car": "BP_BaseCar_C", #for UE4 binary`
    # "motorbike": "MotorBikes_C",#for UE4 binary

    "car":"BP_Hatchback_child_base_C", #for UE5.5 binary
    "motorbike": "BP_BaseBike_C",#for UE5.5 binary
}

# Manual UE asset soft paths (use ** for the prefix segment). Not used with class_name dict above (kept for reference).
asset_path = {
    "player": "	/Game/SmartLocomotion/Blueprints/BP_Character.BP_Character_C",
    "animal": "	/Game/SmartLocomotion/Blueprints/BP_Character.BP_Character_C",
    "drone": "/Game/Drone_Pack/Drone_Bp/BP_drone01.BP_drone01_C",
    "car": [
        "/Game/DD_Vehicles_Advanced/Blueprints/Vehicles/Cars/Hatchback/BP_Hatchback_child_base.BP_Hatchback_child_base_C",
        "/Game/DD_Vehicles_Advanced/Blueprints/Vehicles/Cars/Hatchback/BP_Hatchback_child_extras.BP_Hatchback_child_extras_C",
        "/Game/DD_Vehicles_Advanced/Blueprints/Vehicles/Cars/Hatchback/BP_Hatchback_child_police.BP_Hatchback_child_police_C",
        "/Game/DD_Vehicles_Advanced/Blueprints/Vehicles/Cars/Hatchback/BP_Hatchback_child_taxi.BP_Hatchback_child_taxi_C",
        "/Game/DD_Vehicles_Advanced/Blueprints/Vehicles/Cars/Sedan/BP_Sedan_child_base.BP_Sedan_child_base_C",
        "/Game/DD_Vehicles_Advanced/Blueprints/Vehicles/Cars/Sedan/BP_Sedan_child_extras.BP_Sedan_child_extras_C",
        "/Game/DD_Vehicles_Advanced/Blueprints/Vehicles/Cars/Sedan/BP_Sedan_child_police.BP_Sedan_child_police_C",
        "/Game/DD_Vehicles_Advanced/Blueprints/Vehicles/Cars/Sedan/BP_Sedan_child_taxi.BP_Sedan_child_taxi_C",
        "/Game/DD_Vehicles_Advanced/Blueprints/Vehicles/SUV/BP_SUV_child_base.BP_SUV_child_base_C",
        "/Game/DD_Vehicles_Advanced/Blueprints/Vehicles/SUV/BP_SUV_child_extras.BP_SUV_child_extras_C",
        "/Game/DD_Vehicles_Advanced/Blueprints/Vehicles/SUV/BP_SUV_child_police.BP_SUV_child_police_C",
        "/Game/DD_Vehicles_Advanced/Blueprints/Vehicles/SUV/BP_SUV_child_taxi.BP_SUV_child_taxi_C",
    ],
    "motorbike": [
        "/Game/DD_Vehicles_Advanced/Blueprints/Vehicles/Bikes/Custom/BP_Custom_Base.BP_Custom_Base_C",
        "/Game/DD_Vehicles_Advanced/Blueprints/Vehicles/Bikes/Custom/BP_Custom_Extras.BP_Custom_Extras_C",
        "/Game/DD_Vehicles_Advanced/Blueprints/Vehicles/Bikes/Custom/BP_Custom_Police.BP_Custom_Police_C",
        "/Game/DD_Vehicles_Advanced/Blueprints/Vehicles/Bikes/Enduro/BP_Enduro_Base.BP_Enduro_Base_C",
        "/Game/DD_Vehicles_Advanced/Blueprints/Vehicles/Bikes/Enduro/BP_Enduro_Extras.BP_Enduro_Extras_C",
        "/Game/DD_Vehicles_Advanced/Blueprints/Vehicles/Bikes/Enduro/BP_Enduro_Police.BP_Enduro_Police_C",
        "/Game/DD_Vehicles_Advanced/Blueprints/Vehicles/Bikes/Naked/BP_Naked_Base.BP_Naked_Base_C",
        "/Game/DD_Vehicles_Advanced/Blueprints/Vehicles/Bikes/Naked/BP_Naked_Extras.BP_Naked_Extras_C",
        "/Game/DD_Vehicles_Advanced/Blueprints/Vehicles/Bikes/Naked/BP_Naked_Police.BP_Naked_Police_C",
        "/Game/DD_Vehicles_Advanced/Blueprints/Vehicles/Bikes/BP_BaseBike_TwoPassengers.BP_BaseBike_TwoPassengers_C",
    ],
}

Addition_Vechicles={ #only available in latest UE5.5 package
       "car":["BP_Hatchback_child_extras_C","BP_Hatchback_child_police_C","BP_Hatchback_child_taxi_C",
            "BP_Sedan_child_base_C","BP_Sedan_child_extras_C","BP_Sedan_child_police_C","BP_Sedan_child_taxi_C",
            "BP_SUV_child_base_C","BP_SUV_child_extras_C","BP_SUV_child_police_C","BP_SUV_child_taxi_C"],
    "motorbike":["BP_Custom_Base_C","BP_Custom_Extras_C","BP_Custom_Police_C"
                ,"BP_Enduro_Base_C","BP_Enduro_Extras_C","BP_Enduro_Police_C"
                  ,"BP_Naked_Base_C","BP_Naked_Extras_C","BP_Naked_Police_C"
                  ,"BP_BaseBike_TwoPassengers_C"]
}

player_config = {
        "class_name": [],
        "asset_path": [],
        "internal_nav": True,
        "scale": [1, 1, 1],
        "relative_location": [20, 0, 0],
        "relative_rotation": [0, 0, 0],
        "head_action_continuous": {
            "high": [15, 15, 15],
            "low":  [-15, -15, -15]
        },
        "head_action": [
            [0, 0, 0], [0, 30, 0], [0, -30, 0]],
        "animation_action": ["stand", "jump", "crouch","open_door","enter_vehicle","pickup"],
        "move_action": [
            [0, 100], [0, -100], [15, 50], [-15, 50], [30, 0], [-30, 0], [0, 0]
        ],
        "move_action_continuous": {
            "high": [30, 100],
            "low": [-30, -100]
        }
    }

animal_config = {
        "class_name": [],
        "asset_path": [],
        "internal_nav": True,
        "scale": [1, 1, 1],
        "relative_location": [20, 0, 0],
        "relative_rotation": [0, 0, 0],
        "head_action_continuous": {
            "high": [15, 15, 15],
            "low":  [-15, -15, -15]
        },
        "head_action": [
        [0, 0, 0], [0, 30, 0], [0, -30, 0]],
        "animation_action": ["stand", "jump", "crouch","open_door","enter_vehicle","pickup"],
        "move_action": [
            [0, 200],
            [0, -200],
            [15, 100],
            [-15, 100],
            [30, 0],
            [-30, 0],
            [0, 0]
        ],
        "move_action_continuous": {
            "high": [30, 200],
            "low": [-30, -200]
        }
    }

drone_config = {
        "class_name": [],
        "asset_path": [],
        "internal_nav": False,
        "scale": [0.1, 0.1, 0.1],
        "relative_location": [0, 0, 0],
        "relative_rotation": [0, 0, 0],
        "move_action": [
            [0.5, 0, 0, 0],
            [-0.5, 0, 0, 0],
            [0, 0.5, 0, 0],
            [0, -0.5, 0, 0],
            [0, 0, 0.5, 0],
            [0, 0, -0.5, 0],
            [0, 0, 0, 1],
            [0, 0, 0, -1],
            [0, 0, 0, 0]
        ],
        "move_action_continuous": {
            "high": [1, 1, 1, 1],
            "low": [-1, -1, -1, -1]
        }
        }
car_config = {
        "class_name": [],
        "asset_path": [],
    "internal_nav": True,
    "scale": [1, 1, 1],
    "relative_location": [0, 0,  0],
    "relative_rotation": [0, 0, 0],
    "move_action": [
        [ 1,  0],
        [ -0.3,  0],
        [ 0.5,  1],
        [ 0.5, -1],
        [ 0,  0]
    ],
    "move_action_continuous": {
        "high": [ 1,  1],
        "low":  [0, -1]
    }
}

motorbike_config = {
        "class_name": [],
        "asset_path": [],
    "internal_nav": True,
    "scale": [1, 1, 1],
    "relative_location": [0, 0,  0],
    "relative_rotation": [0, 0, 0],
    "move_action": [
        [1,  0],
        [-0.3,  0],
        [0.5,  1],
        [0.5, -1],
        [0,  0]
    ],
    "move_action_continuous": {
        "high": [1,  1],
        "low":  [0, -1]
    }
}


agents = {
    "player": player_config,
    "animal": animal_config,
    "drone": drone_config,
    "car": car_config,
    "motorbike": motorbike_config
}

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UE5_BACKUP_DIR = os.path.join(_SCRIPT_DIR, 'gym_unrealcv', 'envs', 'setting', 'UE5_backup')


def load_ue5_backup_fields(env_map):
    """Top-level height / reset_area / safe_start from UE5_backup if the file exists."""
    path = os.path.join(UE5_BACKUP_DIR, f'{env_map}.json')
    if not os.path.isfile(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {k: data[k] for k in ('height', 'reset_area', 'safe_start') if k in data}


def fill_agent_class_and_paths(agents_dict, asset_prefix=""):
    """Set class_name (single-element list) and asset_path from module dicts; ** -> asset_prefix."""
    def ap(s):
        if not isinstance(s, str):
            return s
        t = s.strip()
        return t.replace('**', asset_prefix.rstrip('/'), 1) if asset_prefix else t

    for role, cfg in agents_dict.items():
        if role not in class_name:
            continue
        cfg["class_name"] = [class_name[role]]
        apaths = asset_path.get(role)
        if apaths is None:
            cfg["asset_path"] = []
        elif isinstance(apaths, str):
            cfg["asset_path"] = [ap(apaths)]
        else:
            cfg["asset_path"] = [ap(x) for x in apaths]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--env-bin',
        default=None,
        help='Override UE binary for this run; default is os.environ["UnrealEnv"] + platform template (Linux/Win/Mac).',
    )

    parser.add_argument('--env-map', default='all', help='The map to load')
    # parser.add_argument('--target_dir', default='gym_unrealcv/envs/setting/Track', help='The folder to save the json file')
    parser.add_argument('--target_dir', default='gym_unrealcv/envs/setting/Track', help='The folder to save the json file')

    parser.add_argument('--use-docker', action='store_true', help='Run the game in a docker container')
    parser.add_argument('--resolution', '-res', default='640x480', help='The resolution in the unrealcv.ini file')
    parser.add_argument('--display', default=None, help='The display to use')
    parser.add_argument('--use-opengl', action='store_true', help='Use OpenGL for rendering')
    parser.add_argument('--offscreen', action='store_true', help='Use offscreen rendering')
    parser.add_argument('--nullrhi', action='store_true', help='Use the NullRHI')
    parser.add_argument('--show', action='store_true', help='show the get image result')
    parser.add_argument('--gpu-id', default=None, help='The GPU to use')
    parser.add_argument(
        '--asset-prefix',
        default='',
        help='Replaces the ** placeholder in asset_path templates when writing JSON (no trailing slash).',
    )
    args = parser.parse_args()
    asset_prefix = args.asset_prefix or ''
    env_map = args.env_map
    if args.env_map == 'all':
        maps = ['FlexibleRoom',
            'Greek_Island', 'supermarket', 'Brass_Gardens', 'Brass_Palace', 'Brass_Streets',
            'EF_Gus', 'EF_Lewis_1', 'EF_Lewis_2', 'EF_Grounds', 'TemplePlaza', 'Eastern_Garden', 'Western_Garden',
            'Colosseum_Desert',
            'Desert_ruins', 'SchoolGymDay', 'Venice', 'VictorianTrainStation', 'Stadium', 'IndustrialArea',
            'ModularBuilding',
            'DowntownWest', 'TerrainDemo', 'InteriorDemo_NEW', 'AncientRuins', 'Grass_Hills', 'ChineseWaterTown_Ver1',
            'ContainerYard_Night', 'ContainerYard_Day', 'Old_Factory_01', 'racing_track', 'Watermills', 'WildWest',
            'SunsetMap', 'Hospital', 'Medieval_Castle', 'Real_Landscape', 'UndergroundParking', 'Demonstration_Castle',
            'Demonstration_Cave', 'PlatFormHangar', 'PlatformFactory', 'demonstration_BUNKER', 'Arctic',
            'Medieval_Daytime',
            'Medieval_Nighttime', 'ModularGothic_Day', 'ModularGothic_Night',
            'UltimateFarming', 'RuralAustralia_Example_01', 'RuralAustralia_Example_02', 'RuralAustralia_Example_03',
            'LV_Soul_Cave', 'Dungeon_Demo_00', 'SwimmingPool', 'DesertMap', 'RainMap', 'SnowMap',
            'ModularVictorianCity',
            'SuburbNeighborhood_Day','SuburbNeighborhood_Day_dooropen', 'SuburbNeighborhood_Night', 'Storagehouse', 'ModularNeighborhood',
            'ModularSciFiVillage', 'ModularSciFiSeason1', 'LowPolyMedievalInterior_1', 'QA_Holding_Cells_A',
            'ParkingLot', 'Demo_Roof', 'MiddleEast', 'Lighthouse',
            'Cabin_Lake', 'UniversityClassroom', 'Tokyo', 'CommandCenter', 'JapanTrainStation_Optimised',
            'Hotel_Corridor',
            'Museum', 'ForestGasStation',
            'KoreanPalace', 'CourtYard', 'Chinese_Landscape_Demo', 'EnglishCollege', 'OperaHouse', 'AsianTemple',
            'Pyramid', 'PlanetOutDoor',
            'Map_ChemicalPlant_1', 'Hangar',
            'Science_Fiction_valley_town',
            'RussianWinterTownDemo01', 'LookoutTower',
            'LV_Bazaar', 'OperatingRoom',
            'PostSoviet_Village', 'Old_Town', 'AsianMedivalCity', 'StonePineForest', 'TemplesOfCambodia_01_01_Exterior',
            'AbandonedDistrict'
        ]
        env_map = maps[0]
    else:
        maps = [env_map]

    unreal_env = os.environ.get("UnrealEnv")
    if args.env_bin:
        env_bin = args.env_bin
    else:
        if not unreal_env:
            raise SystemExit(
                'Set environment variable UnrealEnv to the root folder that contains the relative binary paths '
                '(same root as in generated JSON), or pass --env-bin explicitly.'
            )
        env_bin = launch_binary_abs(unreal_env, env_bin_template_for_platform(), env_map)

    ue_binary = RunUnreal(ENV_BIN=env_bin, ENV_MAP=env_map)
    env_ip, env_port = ue_binary.start(args.use_docker, parse_resolution(args.resolution), args.display, args.use_opengl, args.offscreen, args.nullrhi, str(args.gpu_id))
    unrealcv = UnrealCv_API(env_port, env_ip, parse_resolution(args.resolution), 'tcp')  # 'tcp' or 'unix', 'unix' is only for local machine in Linux
    # unrealcv.config_ue(parse_res(args.resolution))
    for env_map in maps:
        unrealcv.set_map(env_map)
        time.sleep(5)
        agents = {
            "player": copy.deepcopy(player_config),
            "animal": copy.deepcopy(animal_config),
            "drone": copy.deepcopy(drone_config),
            "car": copy.deepcopy(car_config),
            "motorbike": copy.deepcopy(motorbike_config)
        }
        fill_agent_class_and_paths(agents, asset_prefix)
        env_config = {
            "env_name": None,
            "env_bin": None,
            "env_map": None,
            "env_bin_win": None,
            "env_bin_mac": None,
            "third_cam": {
                "cam_id": 0,
                "pitch": -90,
                "yaw": 0,
                "roll": 0,
                "height_top_view": 1500,
                "fov": 90
            },
            "height": 500,
            "interval": 1000,
            "agents": agents,
            "safe_start": [],
            "reset_area": [0, 0, 0, 0, 0, 0],
            "random_init": False,
            "env": {
                "interactive_door": [],
                "Extra_Vehicles":Addition_Vechicles,
                "Pickable_object":{
                    "class_name": "BP_GrabMoveDrop_C",
                },
                "targets": {
                    "Point": []
                }
            }
        }

        env_config['env_name'] = env_map
        env_config['env_map'] = env_map
        env_config['env_bin'] = format_env_bin_template(ENV_BIN_TEMPLATE_LINUX, env_map)
        env_config['env_bin_win'] = format_env_bin_template(ENV_BIN_TEMPLATE_WIN, env_map)
        env_config['env_bin_mac'] = format_env_bin_template(ENV_BIN_TEMPLATE_MAC, env_map)

        # time.sleep(1)
        cam_num = unrealcv.get_camera_num()
        start_pos_list = []
        # Only need camera 0 for height/reset fallback; syns=False avoids KeyError when
        # get_camera_num() > len(api.cam) (UnrealCV does not pre-register every slot).
        cam_locs = []
        if cam_num > 0:
            cam_locs.append(unrealcv.get_cam_location(0, syns=False))
        # Test the API
        objects = unrealcv.get_objects()

        obj_locations = []
        obj_size = []
        obj_info = {}
        # print(objects)
        print(env_map, 'object number:',len(objects))
        env_config['obj_num'] = len(objects)
        for obj in objects:
            if 'RecastNavMesh' in obj:
                uclass = unrealcv.get_obj_uclass(obj)
                bbox = unrealcv.get_obj_size(obj)
                bbox[0] = bbox[0]/100.0
                bbox[1] = bbox[1]/100.0
                bbox[2] = bbox[2]/100.0
                size = bbox[0] * bbox[1] * bbox[2]
                area = bbox[0] * bbox[1]
                # print(obj, uclass, bbox, size, area)
                env_config['size'] = size
                env_config['area'] = area
                env_config['bbox'] = bbox

        def _ap(s):
            return s.replace('**', asset_prefix.rstrip('/'), 1) if asset_prefix else s

        def match_cam_id(cam_locs, obj_name):
            obj_loc = unrealcv.get_obj_location(obj_name)
            dis_list = []
            for loc in cam_locs:
                distance = unrealcv.get_distance(loc, obj_loc, 3)
                dis_list.append(distance)
            cam_id = dis_list.index(min(dis_list))
            return cam_id
        for obj in objects:
            if re.match(re.compile(r'bp_door', re.I), obj) is not None or re.match(re.compile(r'animateddoor', re.I), obj) is not None:
                env_config['env']['interactive_door'].append(obj)

        env_config['agents'] = agents

        ue5_backup = load_ue5_backup_fields(env_map)
        start_pos_list = [list(cam_locs[0])] if cam_locs else []
        cam_x = [cam_loc[0] for cam_loc in start_pos_list]
        cam_y = [cam_loc[1] for cam_loc in start_pos_list]
        cam_z = [cam_loc[2] for cam_loc in start_pos_list]

        if 'safe_start' in ue5_backup:
            env_config['safe_start'] = ue5_backup['safe_start']
        else:
            env_config['safe_start'] = start_pos_list

        if 'height' in ue5_backup:
            env_config['height'] = ue5_backup['height']
        elif cam_z:
            env_config['height'] = max(cam_z)
        else:
            env_config['height'] = env_config.get('height', 500)

        if 'reset_area' in ue5_backup:
            env_config['reset_area'] = ue5_backup['reset_area']
        elif cam_z:
            env_config['reset_area'] = [min(cam_x), max(cam_x), min(cam_y), max(cam_y), min(cam_z), max(cam_z)]
        else:
            env_config['reset_area'] = [0, 0, 0, 0, 0, 0]

        env_config['third_cam']['height_top_view'] = env_config['height'] + 1000
        # print(env_config)
        if not os.path.exists(args.target_dir):
            os.makedirs(args.target_dir)
        with open(os.path.join(args.target_dir, f'{env_map}.json'), 'w') as json_file:
            json.dump(env_config, json_file, indent=4)


    unrealcv.client.disconnect()
    ue_binary.close()