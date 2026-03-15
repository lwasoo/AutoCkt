import argparse
import os
import sys
from pathlib import Path

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

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from Log import log, LoggerWriter
from autockt.envs.ngspice_vanilla_opamp import TwoStageAmp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", "-cpd", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default="train_ngspice_ppo")
    parser.add_argument("--results_dir", type=str, default=os.path.expanduser("~/ray_results"))
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_gpus", type=int, default=0)
    parser.add_argument("--train_batch_size", type=int, default=1200)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--stop_reward", type=float, default=-0.02)
    parser.add_argument("--stop_iters", type=int, default=200)
    parser.add_argument("--checkpoint_freq", type=int, default=1)
    parser.add_argument("--keep_checkpoints_num", type=int, default=0,
                        help="How many checkpoints to keep. <=0 means keep all.")
    parser.add_argument("--framework", type=str, default="torch", choices=["torch", "tf2"])
    return parser.parse_args()


def build_config(args):
    ppo_config = (
        PPOConfig()
        .environment(
            env="opamp-v0",
            env_config={
                "generalize": True,
                "run_valid": False,
                "sim_cache_size": 20000,
                "max_episode_steps": 30,
            },
            disable_env_checking=True,
        )
        .framework(args.framework)
        .rollouts(
            num_rollout_workers=args.num_workers,
            rollout_fragment_length=args.horizon,
        )
        .training(
            train_batch_size=args.train_batch_size,
            model={"fcnet_hiddens": [64, 64, 64]},
        )
        .resources(num_gpus=args.num_gpus)
    )
    return ppo_config.to_dict()


def main():
    args = parse_args()

    register_env("opamp-v0", lambda config: TwoStageAmp(config))

    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        num_cpus=max(1, args.num_workers + 1),
    )

    sys.stdout = LoggerWriter(log.info)
    sys.stderr = LoggerWriter(log.error)

    try:
        stop_config = {"training_iteration": args.stop_iters}
        if args.stop_reward is not None:
            stop_config["episode_reward_mean"] = args.stop_reward

        keep_ckpt = None if args.keep_checkpoints_num <= 0 else args.keep_checkpoints_num

        run_kwargs = {
            "name": args.experiment_name,
            "config": build_config(args),
            "stop": stop_config,
            "local_dir": args.results_dir,
            "checkpoint_freq": args.checkpoint_freq,
            "checkpoint_at_end": True,
            "keep_checkpoints_num": keep_ckpt,
            "metric": "episode_reward_mean",
            "mode": "max",
            "verbose": 1,
        }

        if args.checkpoint_dir:
            log.info("Restoring from checkpoint: {}".format(args.checkpoint_dir))
            run_kwargs["restore"] = args.checkpoint_dir
        else:
            run_kwargs["resume"] = "AUTO+ERRORED"

        tune.run("PPO", **run_kwargs)

    except Exception as e:
        log.error("Error occurred: {}".format(e))
        raise
    finally:
        log.info("Training completed.")
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        ray.shutdown()


if __name__ == "__main__":
    main()
