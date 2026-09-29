#!/usr/bin/env python3
"""A/B reset latency and deterministic replay in an existing local UE editor."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
from ue_go1_env import UEGo1Pool, numbers, state_from

FIELDS = ('obs', 'control_targets', 'sim_time', 'foot_positions', 'foot_velocities', 'foot_contacts')


def differences(reference, actual):
    result = {}
    missing = [name for name in FIELDS if name not in reference or name not in actual]
    if missing:
        raise RuntimeError(f'Missing replay fields: {missing}')
    for name in FIELDS:
        if name in reference:
            a, b = np.asarray(reference[name], dtype=float), np.asarray(actual[name], dtype=float)
            if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                raise RuntimeError(f'Invalid {name} in replay')
            difference = float(np.max(np.abs(a - b))) if a.size else 0.
            if difference > 1e-6:
                raise RuntimeError(f'Reset/replay differs in {name}: {difference}')
            result[name] = difference
    return result


def save(path, result):
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--connect', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cycles', type=int, default=8, help='Number of alternating pairs')
    parser.add_argument('--steps', type=int, default=24)
    args = parser.parse_args()
    if args.cycles < 2 or args.steps < 1:
        parser.error('At least two cycles and one step are required')
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = {'status': 'running', 'scope': 'two isolated replicas, one active robot; reset A/B, not PPO',
              'cycles': args.cycles, 'steps_per_episode': args.steps, 'records': [],
              'control_dt': .02, 'implementation_sha256': {
                  name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                  for name in ('benchmark_go1_reset.py', 'ue_go1_env.py')}}
    save(output / 'results.json', result)
    pool = None
    try:
        pool = UEGo1Pool(None, 1, 2, 23900, output / 'pool', request_timeout=120,
                        reset_mode='auto', connect=args.connect)
        pool.reset()
        worker = pool.workers[0]
        robot, peer = worker.robots
        if not all(r.reset_api_supported for r in worker.robots):
            raise RuntimeError('This comparison requires reset API v2 in the running UE binary')
        result['game_context'] = worker.world_context
        result['initial_states'] = pool.states
        result['initial_spawn'] = robot.location.tolist()
        result['initial_metrics'] = pool.metrics()
        model_path = Path(robot.asset_contract['runtime_mjcf'])
        peer_initial = state_from(worker.request(f'vget /object/{peer.name}/mujoco_go1_policy_obs'))
        initial = robot.state
        reference = None
        actions = [0.1 * np.sin(step * .2 + np.arange(12)) for step in range(args.steps)]
        for cycle in range(args.cycles):
            modes = ('rebuild', 'auto') if cycle % 2 == 0 else ('auto', 'rebuild')
            for mode in modes:
                before_mtime = model_path.stat().st_mtime_ns
                before_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
                before_requests = worker.request_seconds
                start = time.perf_counter()
                pool.reset_indices([0], reset_mode=mode)
                reset_wall = time.perf_counter() - start
                reset_rpc = worker.request_seconds - before_requests
                reset_state = robot.state
                expected = 'reused' if mode == 'auto' else 'rebuilt'
                if reset_state['last_reset_result'] != expected:
                    raise RuntimeError(f'{mode} returned {reset_state["last_reset_result"]}, expected {expected}')
                reset_errors = differences(initial, reset_state)
                after_mtime = model_path.stat().st_mtime_ns
                after_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
                if mode == 'auto' and (before_mtime != after_mtime or before_hash != after_hash):
                    raise RuntimeError('auto reset unexpectedly rewrote the runtime model')
                if mode == 'rebuild' and before_mtime == after_mtime:
                    raise RuntimeError('rebuild did not regenerate the runtime model')
                if before_hash != after_hash:
                    raise RuntimeError('Same-spawn rebuild produced a different runtime model')
                trajectory = []
                sample_start = time.perf_counter()
                for action in actions:
                    previous_time = float(robot.state['sim_time'])
                    robot.state = state_from(worker.request(
                        f'vset /object/{robot.name}/mujoco_go1_policy_step {numbers(action)}'), previous_time)
                    trajectory.append(robot.state)
                sample_wall = time.perf_counter() - sample_start
                if reference is None:
                    reference = trajectory
                replay_error = max(max(differences(a, b).values(), default=0.)
                                   for a, b in zip(reference, trajectory))
                peer_after = state_from(worker.request(f'vget /object/{peer.name}/mujoco_go1_policy_obs'))
                peer_error = max(differences(peer_initial, peer_after).values(), default=0.)
                record = {'cycle': cycle, 'mode': mode, 'result': expected,
                          'reset_wall_seconds': reset_wall, 'reset_rpc_seconds': reset_rpc,
                          'sampling_seconds': sample_wall, 'steps': args.steps,
                          'reset_max_difference': max(reset_errors.values(), default=0.),
                          'trajectory_max_difference': replay_error, 'peer_max_difference': peer_error,
                          'runtime_mjcf_sha256': after_hash, 'model_rewritten': before_mtime != after_mtime}
                result['records'].append(record)
                save(output / 'results.json', result)
                print(json.dumps(record), flush=True)
        summary = {}
        for mode in ('auto', 'rebuild'):
            records = [r for r in result['records'] if r['mode'] == mode]
            rpc = [r['reset_rpc_seconds'] for r in records]
            wall = [r['reset_wall_seconds'] for r in records]
            sampling = sum(r['sampling_seconds'] for r in records)
            transitions = args.steps * len(records)
            summary[mode] = {'count': len(records), 'reset_rpc_mean_ms': statistics.mean(rpc)*1000,
                             'reset_rpc_median_ms': statistics.median(rpc)*1000,
                             'reset_rpc_p95_ms': float(np.percentile(rpc, 95))*1000,
                             'reset_wall_mean_ms': statistics.mean(wall)*1000,
                             'sampling_steps_per_second': transitions/sampling,
                             'fixed_episode_steps_per_second_including_reset': transitions/(sampling+sum(wall))}
        result.update(status='passed', summary=summary,
                      reset_rpc_mean_speedup=summary['rebuild']['reset_rpc_mean_ms']/summary['auto']['reset_rpc_mean_ms'],
                      final_metrics=pool.metrics())
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        if pool is not None:
            pool.close()
            result['cleanup_errors'] = pool.cleanup_errors
            if pool.cleanup_errors:
                result['status'] = 'failed_cleanup'
        save(output / 'results.json', result)
    print(json.dumps({'status': result['status'], 'summary': result['summary']}, indent=2))
    return 0 if result['status'] == 'passed' else 1

if __name__ == '__main__':
    raise SystemExit(main())
