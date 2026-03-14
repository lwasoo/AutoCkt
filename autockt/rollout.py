#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Log import log

import argparse
import json
import os
import pickle
import numpy as np

import gym
import ray
from ray.rllib.agents.registry import get_agent_class
from ray.tune.registry import register_env

from autockt.envs.ngspice_vanilla_opamp import TwoStageAmp

EXAMPLE_USAGE = """
Example Usage via RLlib CLI:
    rllib rollout /tmp/ray/checkpoint_dir/checkpoint-0 --run DQN
    --env CartPole-v0 --steps 1000000 --out rollouts.pkl

Example Usage via executable:
    ./rollout.py /tmp/ray/checkpoint_dir/checkpoint-0 --run DQN
    --env CartPole-v0 --steps 1000000 --out rollouts.pkl
"""

register_env("opamp-v0", lambda config: TwoStageAmp(config))


def create_parser(parser_creator=None):
    parser_creator = parser_creator or argparse.ArgumentParser
    parser = parser_creator(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Roll out a reinforcement learning agent given a checkpoint.",
        epilog=EXAMPLE_USAGE,
    )
    parser.add_argument("checkpoint", type=str, help="Checkpoint from which to roll out.")
    required_named = parser.add_argument_group("required named arguments")
    required_named.add_argument(
        "--run",
        type=str,
        required=True,
        help=(
            "The algorithm or model to train. This may refer to the name "
            "of a built-on algorithm (e.g. RLLib's DQN or PPO), or a "
            "user-defined trainable function or class registered in the tune registry."
        ),
    )
    required_named.add_argument("--env", type=str, help="The gym environment to use.")
    parser.add_argument(
        "--no-render",
        default=False,
        action="store_const",
        const=True,
        help="Suppress rendering of the environment.",
    )
    parser.add_argument("--steps", default=10000, help="Number of steps to roll out.")
    parser.add_argument("--out", default=None, help="Output filename.")
    parser.add_argument(
        "--config",
        default="{}",
        type=json.loads,
        help=(
            "Algorithm-specific configuration (e.g. env, hyperparams). "
            "Suppresses loading of configuration from checkpoint."
        ),
    )
    parser.add_argument(
        "--num_val_specs",
        type=int,
        default=50,
        help="Number of untrained objectives to test on",
    )
    parser.add_argument(
        "--traj_len",
        type=int,
        default=60,
        help="Length of each trajectory",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=10,
        help="Persist intermediate pickle files every N episodes (0 means only final save).",
    )
    return parser


def run(args, parser):
    config = args.config
    if not config:
        config_dir = os.path.dirname(args.checkpoint)
        config_path = os.path.join(config_dir, "params.json")
        if not os.path.exists(config_path):
            config_path = os.path.join(config_dir, "../params.json")
        if not os.path.exists(config_path):
            raise ValueError(
                "Could not find params.json in either the checkpoint dir or its parent directory."
            )
        with open(config_path) as f:
            config = json.load(f)
        if "num_workers" in config:
            config["num_workers"] = 0

    if not args.env:
        if not config.get("env"):
            parser.error("the following arguments are required: --env")
        args.env = config.get("env")

    ray.init()

    cls = get_agent_class(args.run)
    agent = cls(env=args.env, config=config)
    agent.restore(args.checkpoint)

    rollout(
        agent=agent,
        env_name=args.env,
        num_val_specs=args.num_val_specs,
        traj_len=args.traj_len,
        out=args.out,
        no_render=args.no_render,
        save_every=max(0, args.save_every),
    )


def unlookup(norm_spec, goal_spec):
    spec = -1 * np.multiply((norm_spec + 1), goal_spec) / (norm_spec - 1)
    return spec


def _dump_pickle(path, payload):
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def rollout(agent, env_name, num_val_specs, traj_len, out=None, no_render=True, save_every=10):
    if hasattr(agent, "local_evaluator"):
        env_config = {
            "generalize": True,
            "num_valid": num_val_specs,
            "save_specs": False,
            "run_valid": True,
        }
        if env_name == "opamp-v0":
            env = TwoStageAmp(env_config=env_config)
        else:
            env = gym.make(env_name)
    else:
        env = gym.make(env_name)

    norm_spec_ref = env.global_g
    spec_num = len(env.specs)

    if hasattr(agent, "local_evaluator"):
        state_init = agent.local_evaluator.policy_map["default"].get_initial_state()
    else:
        state_init = []
    use_lstm = bool(state_init)

    log_each_step = False

    rollouts = []
    next_states = []
    obs_reached = []
    obs_nreached = []
    action_arr_comp = []
    rollout_steps = 0
    reached_spec = 0

    while rollout_steps < num_val_specs:
        action_array = []
        rollout_num = [] if out is not None else None
        state = env.reset()

        done = False
        reward_total = 0.0
        steps = 0
        while not done and steps < traj_len:
            if use_lstm:
                action, state_init, _ = agent.compute_action(state, state=state_init)
            else:
                action = agent.compute_action(state)
            action_array.append(action)

            next_state, reward, done, _ = env.step(action)
            if log_each_step:
                log_details = "action: {} | reward: {} | done: {}".format(action, reward, done)
                log.info(log_details)
            reward_total += reward
            if not no_render:
                env.render()
            if rollout_num is not None:
                rollout_num.append(reward)
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

        if rollout_num is not None:
            rollouts.append(rollout_num)

        rollout_steps += 1
        log.info("Episode reward: {}".format(reward_total))
        log.info("Specs reached: {}/{}".format(str(reached_spec), str(rollout_steps)))

        should_save = save_every > 0 and (rollout_steps % save_every == 0)
        if should_save:
            _dump_pickle("action_arr_test", action_arr_comp)
            _dump_pickle("opamp_obs_reached_test", obs_reached)
            _dump_pickle("opamp_obs_nreached_test", obs_nreached)

    _dump_pickle("action_arr_test", action_arr_comp)
    _dump_pickle("opamp_obs_reached_test", obs_reached)
    _dump_pickle("opamp_obs_nreached_test", obs_nreached)

    if out is not None:
        _dump_pickle(out, {"rewards": rollouts, "next_states": next_states})

    log.info("Num specs reached: {}/{}".format(str(reached_spec), str(num_val_specs)))


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    run(args, parser)
