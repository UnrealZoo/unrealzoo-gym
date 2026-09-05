"""Compare lit and lit_shared in a minimal multi-agent control loop.

The timed loop only contains random actions, agent control commands and one
image from every agent camera. It does not call env.step, so pose queries,
reward calculation and the tracking mask are excluded.
"""

import argparse
import os
import time

import cv2
import gym
import numpy as np
from unrealcv import SharedCommand

import gym_unrealcv
from gym_unrealcv.envs.wrappers import augmentation, configUE


def get_control_cmds(env, actions):
    """Use the same action mapping and control commands as BaseEnv.step."""
    moves, turns, animations = env.action_mapping(actions, env.player_list)

    move_cmds = [
        env.unrealcv.set_move_bp(obj, moves[i], return_cmd=True)
        for i, obj in enumerate(env.player_list)
        if moves[i] is not None
    ]
    head_cmds = [
        env.unrealcv.set_cam(
            obj,
            env.agents[obj]['relative_location'],
            turns[i],
            return_cmd=True,
        )
        for i, obj in enumerate(env.player_list)
        if turns[i] is not None
    ]
    anim_cmds = [
        env.unrealcv.set_animation(obj, animations[i], return_cmd=True)
        for i, obj in enumerate(env.player_list)
        if animations[i] is not None
    ]
    return move_cmds + head_cmds + anim_cmds


def decode_bmp(data):
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError('Failed to decode lit bmp')
    return image


def minimal_step(env, cam_ids, transport):
    """Random actions -> control batch -> two-camera image batch."""
    actions = [space.sample() for space in env.action_space]
    control_cmds = get_control_cmds(env, actions)
    if control_cmds:
        env.unrealcv.batch_cmd(control_cmds, None)

    if transport == 'lit':
        image_cmds = [f'vget /camera/{cam_id}/lit bmp' for cam_id in cam_ids]
    else:
        # UnrealCV 1.3.2 keeps SharedCommand objects in one wire batch. Earlier
        # clients silently fell back to one request per camera here.
        image_cmds = [
            SharedCommand(f'vget /camera/{cam_id}/lit_shared', 'bmp')
            for cam_id in cam_ids
        ]
    images = [decode_bmp(data) for data in env.unrealcv.client.request(image_cmds)]

    env.count_steps += 1
    return images


def benchmark(env, transport, warmup_steps, test_steps, rounds):
    cam_ids = env.cam_list[:len(env.player_list)]
    if len(cam_ids) != len(env.player_list) or any(cam_id < 0 for cam_id in cam_ids):
        raise RuntimeError(f'Invalid camera ids: {cam_ids}')

    for _ in range(warmup_steps):
        minimal_step(env, cam_ids, transport)

    round_fps = []
    for round_id in range(rounds):
        start = time.perf_counter()
        for _ in range(test_steps):
            minimal_step(env, cam_ids, transport)
        fps = test_steps / (time.perf_counter() - start)
        round_fps.append(fps)
        print(f'{transport} round {round_id + 1}: {fps:.2f} step FPS, '
              f'{fps * len(cam_ids):.2f} image FPS')

    return float(np.mean(round_fps)), float(np.std(round_fps))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark lit and lit_shared with two agents')
    parser.add_argument('-e', '--env-id', default='UnrealTrack-FlexibleRoom-ContinuousColor-v0')
    parser.add_argument('--num-agents', type=int, default=2)
    parser.add_argument('--warmup-steps', type=int, default=20)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--offscreen', action='store_true')
    args = parser.parse_args()

    if not os.environ.get('UnrealEnv'):
        parser.error('Set UnrealEnv to the directory containing the packaged environment first')

    env = gym.make(args.env_id)
    env = configUE.ConfigUEWrapper(
        env,
        offscreen=args.offscreen,
        resolution=(640, 640),
    )
    env.unwrapped.agents_category = ['player']
    env = augmentation.RandomPopulationWrapper(
        env,
        args.num_agents,
        args.num_agents,
        random_target=False,
    )

    results = {}
    for transport in ['lit', 'lit_shared']:
        np.random.seed(args.seed)
        env.seed(args.seed)
        env.reset()  # Reset and initialization are outside the timed loop.

        print(f'\ntransport: {transport}')
        print(f'agents: {env.unwrapped.player_list}')
        print(f'cameras: {env.unwrapped.cam_list[:args.num_agents]}')
        results[transport] = benchmark(
            env.unwrapped,
            transport,
            args.warmup_steps,
            args.steps,
            args.rounds,
        )
    env.close()

    lit_fps, lit_std = results['lit']
    shared_fps, shared_std = results['lit_shared']
    print('\n=== 2-agent 640x640 minimal control loop ===')
    print(f'lit:        {lit_fps:.2f} +/- {lit_std:.2f} step FPS')
    print(f'lit_shared: {shared_fps:.2f} +/- {shared_std:.2f} step FPS')
    print(f'speedup:    {shared_fps / lit_fps:.2f}x')
