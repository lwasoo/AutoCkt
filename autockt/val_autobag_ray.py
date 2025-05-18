import os

from Log import log, LoggerWriter
import ray
import ray.tune as tune
from ray.rllib.agents import ppo
from autockt.envs.ngspice_vanilla_opamp import TwoStageAmp

import argparse
import sys
import io

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_dir', '-cpd', type=str)
args = parser.parse_args()
ray.init()

# 软连接目标路径
ray_results_target = os.path.expanduser("~/workspace310a/ray_results")
ray_results_link = os.path.expanduser("~/ray_results")

try:
    if not os.path.exists(ray_results_target):
        os.makedirs(ray_results_target)
        log.info("Created directory: {}".format(ray_results_target))

    # 确保 ~/ray_results 不是目录或已有文件
    if os.path.exists(ray_results_link) or os.path.islink(ray_results_link):
        os.remove(ray_results_link)  # 先删除原有的 ~/ray_results（如果存在）

    # 创建软链接
    os.symlink(ray_results_target, ray_results_link)
    log.info("Created symlink: {} -> {}".format(ray_results_link, ray_results_target))

except Exception as e:
    log.error("Failed to create symlink: {}".format(e))

sys.stdout = LoggerWriter(log.info)  # 所有 print() 变成 log.info()
sys.stderr = LoggerWriter(log.error)  # 捕获错误信息

try:
    # configures training of the agent with associated hyperparameters
    # See Ray documentation for details on each parameter
    config_train = {
        # "sample_batch_size": 200,
        "train_batch_size": 1200,
        # "sgd_minibatch_size": 1200,
        # "num_sgd_iter": 3,
        # "lr":1e-3,
        # "vf_loss_coeff": 0.5,
        "horizon": 30,
        "num_gpus": 0,
        "model": {"fcnet_hiddens": [50, 50, 50]},
        "num_workers": 6,
        "env_config": {"generalize": True, "run_valid": False},
    }

    # Runs training and saves the result in ~/ray_results/train_ngspice_45nm
    # If checkpoint fails for any reason, training can be restored
    if not args.checkpoint_dir:
        trials = tune.run_experiments({
            "train_45nm_ngspice": {
                "checkpoint_freq": 1,
                "run": "PPO",
                "env": TwoStageAmp,
                "stop": {"episode_reward_mean": -0.02},
                "config": config_train},
        })
    else:
        log.info("RESTORING NOW!!!!!!")
        tune.run_experiments({
            "restore_ppo": {
                "run": "PPO",
                "config": config_train,
                "env": TwoStageAmp,
                # "restore": trials[0]._checkpoint.value},
                "restore": args.checkpoint_dir,
                "stop": {"episode_reward_mean": -0.02},
                "checkpoint_freq": 1},
        })

except Exception as e:
    log.error("Error occurred: {}".format(e))

finally:
    log.info("Training completed.")
    sys.stdout = sys.__stdout__  # 恢复正常输出
    sys.stderr = sys.__stderr__
