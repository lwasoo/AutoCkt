#!/usr/bin/env python

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import pkg_resources
    import packaging
    if not hasattr(pkg_resources, "packaging"):
        pkg_resources.packaging = packaging
except Exception:
    pass

try:
    import gymnasium as gym
except ImportError:
    import gym

import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env

from Log import log
from autockt.envs.ngspice_vanilla_opamp import TwoStageAmp


def create_parser():
    parser = argparse.ArgumentParser(description="Roll out a trained reinforcement learning checkpoint.")
    parser.add_argument("checkpoint", type=str, help="Ray RLlib checkpoint path.")
    parser.add_argument("--env", type=str, default="opamp-v0", help="Gym environment name.")
    parser.add_argument("--num_val_specs", type=int, default=50, help="Number of untrained objectives to test on.")
    parser.add_argument("--traj_len", type=int, default=60, help="Length of each trajectory.")
    parser.add_argument("--save_every", type=int, default=10, help="Persist intermediate outputs every N episodes.")
    parser.add_argument("--out", default=None, help="Optional output pickle for reward trajectories.")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering.")
    return parser


def unlookup(norm_spec, goal_spec):
    return -1 * np.multiply((norm_spec + 1), goal_spec) / (norm_spec - 1)


def _dump_pickle(path, payload):
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def _reset_env(env):
    reset_out = env.reset()
    if isinstance(reset_out, tuple) and len(reset_out) == 2:
        return reset_out[0]
    return reset_out


def _step_env(env, action):
    out = env.step(action)
    if len(out) == 5:
        next_obs, reward, terminated, truncated, info = out
        done = terminated or truncated
        return next_obs, reward, done, info
    next_obs, reward, done, info = out
    return next_obs, reward, done, info


def build_env(env_name, num_val_specs):
    if env_name == "opamp-v0":
        env_config = {
            "generalize": True,
            "num_valid": num_val_specs,
            "save_specs": False,
            "run_valid": True,
        }
        return TwoStageAmp(env_config=env_config)
    return gym.make(env_name)


def rollout(algo, env_name, num_val_specs, traj_len, out=None, no_render=True, save_every=10):
    env = build_env(env_name, num_val_specs)

    norm_spec_ref = env.global_g
    spec_num = len(env.specs)

    rollouts = []
    next_states = []
    obs_reached = []
    obs_nreached = []
    action_arr_comp = []
    reached_spec = 0

    for rollout_idx in range(1, num_val_specs + 1):
        action_array = []
        rollout_rewards = [] if out is not None else None
        state = _reset_env(env)

        done = False
        reward_total = 0.0
        steps = 0

        while not done and steps < traj_len:
            action = algo.compute_single_action(state, explore=False)
            action_array.append(action)

            next_state, reward, done, _ = _step_env(env, action)
            reward_total += reward

            if not no_render:
                env.render()

            if rollout_rewards is not None:
                rollout_rewards.append(reward)
                next_states.append(next_state)

            steps += 1
            state = next_state

        norm_ideal_spec = state[spec_num:spec_num + spec_num]
        ideal_spec = unlookup(norm_ideal_spec, norm_spec_ref)

        if done:
            reached_spec += 1
            obs_reached.append(ideal_spec)
            action_arr_comp.append(action_array)
        else:
            obs_nreached.append(ideal_spec)

        if rollout_rewards is not None:
            rollouts.append(rollout_rewards)

        log.info("Episode reward: {}".format(reward_total))
        log.info("Specs reached: {}/{}".format(reached_spec, rollout_idx))

        should_save = save_every > 0 and (rollout_idx % save_every == 0)
        if should_save:
            _dump_pickle("action_arr_test", action_arr_comp)
            _dump_pickle("opamp_obs_reached_test", obs_reached)
            _dump_pickle("opamp_obs_nreached_test", obs_nreached)

    _dump_pickle("action_arr_test", action_arr_comp)
    _dump_pickle("opamp_obs_reached_test", obs_reached)
    _dump_pickle("opamp_obs_nreached_test", obs_nreached)

    if out is not None:
        _dump_pickle(out, {"rewards": rollouts, "next_states": next_states})

    log.info("Num specs reached: {}/{}".format(reached_spec, num_val_specs))


if __name__ == "__main__":
    args = create_parser().parse_args()
    register_env("opamp-v0", lambda config: TwoStageAmp(config))
    ray.init(ignore_reinit_error=True, include_dashboard=False)
    try:
        algo = Algorithm.from_checkpoint(args.checkpoint)
        rollout(
            algo=algo,
            env_name=args.env,
            num_val_specs=args.num_val_specs,
            traj_len=args.traj_len,
            out=args.out,
            no_render=args.no_render,
            save_every=max(0, args.save_every),
        )
    finally:
        ray.shutdown()
