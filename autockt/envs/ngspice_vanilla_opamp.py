"""
A new ckt environment based on a new structure of MDP
"""
from Log import log

import gym
from gym import spaces

import numpy as np
import random
from collections import OrderedDict
from pathlib import Path
import yaml
import pickle

from eval_engines.ngspice.TwoStageClass import *
from autockt.envs.read_yaml import OrderedDictYAMLLoader


class TwoStageAmp(gym.Env):
    metadata = {'render.modes': ['human']}

    PERF_LOW = -1
    PERF_HIGH = 0

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    CIR_YAML = str(
        PROJECT_ROOT / "eval_engines" / "ngspice" / "ngspice_inputs" / "yaml_files" / "two_stage_opamp.yaml"
    )

    def __init__(self, env_config):
        self.multi_goal = env_config.get("multi_goal", False)
        self.generalize = env_config.get("generalize", False)
        num_valid = env_config.get("num_valid", 50)
        self.specs_save = env_config.get("save_specs", False)
        self.valid = env_config.get("run_valid", False)
        self.done_reward = float(env_config.get("done_reward", 10.0))
        self.reward_tolerance = float(env_config.get("reward_tolerance", -0.02))
        self.sim_cache_size = int(env_config.get("sim_cache_size", 20000))

        self.env_steps = 0
        with open(TwoStageAmp.CIR_YAML, 'r') as f:
            yaml_data = yaml.load(f, OrderedDictYAMLLoader)

        if self.generalize is False:
            specs = yaml_data['target_specs']
        else:
            load_specs_path = TwoStageAmp.PROJECT_ROOT / "autockt" / "gen_specs" / "ngspice_specs_gen_two_stage_opamp"
            with open(str(load_specs_path), 'rb') as f:
                specs = pickle.load(f)

        self.specs = OrderedDict(sorted(specs.items(), key=lambda k: k[0]))
        if self.specs_save:
            with open("specs_" + str(num_valid) + str(random.randint(1, 100000)), 'wb') as f:
                pickle.dump(self.specs, f)

        self.specs_ideal = []
        self.specs_id = list(self.specs.keys())
        self.fixed_goal_idx = -1
        self.num_os = len(list(self.specs.values())[0])

        params = yaml_data['params']
        self.params = []
        self.params_id = list(params.keys())

        for value in params.values():
            param_vec = np.arange(value[0], value[1], value[2])
            self.params.append(param_vec)

        self.sim_env = TwoStageClass(yaml_path=TwoStageAmp.CIR_YAML, num_process=1, path=str(TwoStageAmp.PROJECT_ROOT))
        self.action_meaning = env_config.get("action_meaning", [-1, 0, 2])
        self.action_space = spaces.Tuple([spaces.Discrete(len(self.action_meaning))] * len(self.params_id))

        param_low = [0] * len(self.params_id)
        param_high = [(len(param_vec) - 1) for param_vec in self.params]
        self.observation_space = spaces.Box(
            low=np.array([TwoStageAmp.PERF_LOW] * 2 * len(self.specs_id) + param_low, dtype=np.float32),
            high=np.array([TwoStageAmp.PERF_HIGH] * 2 * len(self.specs_id) + param_high, dtype=np.float32),
            dtype=np.float32,
        )

        self.cur_specs = np.zeros(len(self.specs_id), dtype=np.float32)
        self.cur_params_idx = np.zeros(len(self.params_id), dtype=np.int32)

        self.sim_cache = {}
        default_init = [33, 33, 33, 33, 33, 14, 20]
        init_param_idx = env_config.get("init_param_idx", default_init)
        if len(init_param_idx) != len(self.params_id):
            init_param_idx = [len(param_vec) // 2 for param_vec in self.params]
        self.init_param_idx = np.array(init_param_idx, dtype=np.int32)

        self.global_g = []
        for spec in list(self.specs.values()):
            self.global_g.append(float(spec[self.fixed_goal_idx]))
        self.g_star = np.array(self.global_g)
        self.global_g = np.array(yaml_data['normalize'])

        self.obj_idx = 0

    def reset(self):
        if self.generalize or self.multi_goal:
            if self.generalize and self.valid:
                if self.obj_idx > self.num_os - 1:
                    self.obj_idx = 0
                idx = self.obj_idx
                self.obj_idx += 1
            else:
                idx = random.randint(0, self.num_os - 1)
            self.specs_ideal = np.array([spec[idx] for spec in self.specs.values()])
        else:
            self.specs_ideal = self.g_star

        self.specs_ideal_norm = self.lookup(self.specs_ideal, self.global_g)

        self.cur_params_idx = self.init_param_idx.copy()
        self.cur_specs = self.update(self.cur_params_idx)
        cur_spec_norm = self.lookup(self.cur_specs, self.global_g)

        self.ob = np.concatenate([cur_spec_norm, self.specs_ideal_norm, self.cur_params_idx]).astype(np.float32)
        return self.ob

    def step(self, action):
        """
        :param action: is vector with elements between 0 and 1 mapped to the index of the corresponding parameter
        :return:
        """

        action = list(np.reshape(np.array(action), (np.array(action).shape[0],)))
        self.cur_params_idx = self.cur_params_idx + np.array([self.action_meaning[a] for a in action])

        self.cur_params_idx = np.clip(
            self.cur_params_idx,
            [0] * len(self.params_id),
            [(len(param_vec) - 1) for param_vec in self.params],
        )
        self.cur_specs = self.update(self.cur_params_idx)
        cur_spec_norm = self.lookup(self.cur_specs, self.global_g)
        reward = self.reward(self.cur_specs, self.specs_ideal)
        done = reward >= self.done_reward

        if done:
            log_details = (
                "\n{0}\n"
                "params = {1}\n"
                "specs: {2}\n"
                "ideal specs: {3}\n"
                "re: {4}\n"
                "{0}"
            ).format('-' * 10, self.cur_params_idx, self.cur_specs, self.specs_ideal, reward)
            log.info(log_details)

        self.ob = np.concatenate([cur_spec_norm, self.specs_ideal_norm, self.cur_params_idx]).astype(np.float32)
        self.env_steps += 1

        return self.ob, reward, done, {}

    def lookup(self, spec, goal_spec):
        goal_spec = np.array([float(e) for e in goal_spec], dtype=np.float64)
        spec = np.array(spec, dtype=np.float64)
        den = goal_spec + spec
        den = np.where(np.abs(den) < 1e-12, 1e-12, den)
        norm_spec = (spec - goal_spec) / den
        return norm_spec

    def reward(self, spec, goal_spec):
        """
        Reward: doesn't penalize for overshooting spec, is negative
        """
        rel_specs = self.lookup(spec, goal_spec)
        reward = 0.0
        for i, rel_spec in enumerate(rel_specs):
            if self.specs_id[i] == 'ibias_max':
                rel_spec = rel_spec * -1.0
            if rel_spec < 0:
                reward += rel_spec

        return reward if reward < self.reward_tolerance else self.done_reward

    def update(self, params_idx):
        """
        :param params_idx: parameter index array
        :return: simulated specs array
        """
        cache_key = tuple(int(x) for x in params_idx)
        if cache_key in self.sim_cache:
            return self.sim_cache[cache_key]

        params = [self.params[i][params_idx[i]] for i in range(len(self.params_id))]
        param_val = [OrderedDict(list(zip(self.params_id, params)))]

        _, specs_dict, info = self.sim_env.create_design_and_simulate(param_val[0])
        if info != 0 or specs_dict is None:
            cur_specs = np.zeros(len(self.specs_id), dtype=np.float64)
        else:
            cur_specs = OrderedDict(sorted(specs_dict.items(), key=lambda k: k[0]))
            cur_specs = np.array(list(cur_specs.values()), dtype=np.float64)

        self.sim_cache[cache_key] = cur_specs
        if len(self.sim_cache) > self.sim_cache_size:
            self.sim_cache.pop(next(iter(self.sim_cache)))

        return cur_specs


def main():
    env_config = {"generalize": True, "valid": True}
    env = TwoStageAmp(env_config)
    env.reset()
    env.step([2, 2, 2, 2, 2, 2, 2])


if __name__ == "__main__":
    main()
