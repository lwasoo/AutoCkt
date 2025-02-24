"""
A new ckt environment based on a new structure of MDP
"""
from Log import log
from func_decorator import debug_log

import gym  #RL的主要环境
from gym import spaces

import numpy as np
import random
import psutil

from multiprocessing.dummy import Pool as ThreadPool
from collections import OrderedDict
import yaml
import yaml.constructor
import statistics
import IPython
import itertools
from eval_engines.util.core import *
import pickle
import os

from eval_engines.ngspice.TwoStageClass import *
from autockt.envs.read_yaml import OrderedDictYAMLLoader


class TwoStageAmp(gym.Env): #该类继承子gym.Env
    metadata = {'render.modes': ['human']}

    PERF_LOW = -1
    PERF_HIGH = 0

    # obtains yaml file
    path = os.getcwd()
    CIR_YAML = path + '/eval_engines/ngspice/ngspice_inputs/yaml_files/two_stage_opamp.yaml'

    def __init__(self, env_config):
        self.multi_goal = env_config.get("multi_goal", False)
        self.generalize = env_config.get("generalize", False)
        num_valid = env_config.get("num_valid", 50)
        self.specs_save = env_config.get("save_specs", False)
        self.valid = env_config.get("run_valid", False)

        self.env_steps = 0
        with open(TwoStageAmp.CIR_YAML, 'r') as f:
            yaml_data = yaml.load(f, OrderedDictYAMLLoader)

        # design specs
        if self.generalize == False:
            specs = yaml_data['target_specs']
        else:
            load_specs_path = TwoStageAmp.path + "/autockt/gen_specs/ngspice_specs_gen_two_stage_opamp"
            with open(load_specs_path, 'rb') as f:
                specs = pickle.load(f)

        self.specs = OrderedDict(sorted(specs.items(), key=lambda k: k[0]))
        if self.specs_save:
            with open("specs_" + str(num_valid) + str(random.randint(1, 100000)), 'wb') as f:
                pickle.dump(self.specs, f)

        self.specs_ideal = []
        self.specs_id = list(self.specs.keys())
        self.fixed_goal_idx = -1
        self.num_os = len(list(self.specs.values())[0])

        # param array
        params = yaml_data['params']
        self.params = []
        self.params_id = list(params.keys())

        for value in params.values():
            param_vec = np.arange(value[0], value[1], value[2])
            self.params.append(param_vec)

        # initialize sim environment
        self.sim_env = TwoStageClass(yaml_path=TwoStageAmp.CIR_YAML, num_process=1, path=TwoStageAmp.path)
        self.action_meaning = [-1, 0, 2]
        self.action_space = spaces.Tuple([spaces.Discrete(len(self.action_meaning))] * len(self.params_id))
        # self.action_space = spaces.Discrete(len(self.action_meaning)**len(self.params_id))
        self.observation_space = spaces.Box(
            low=np.array([TwoStageAmp.PERF_LOW] * 2 * len(self.specs_id) + len(self.params_id) * [1]),
            high=np.array([TwoStageAmp.PERF_HIGH] * 2 * len(self.specs_id) + len(self.params_id) * [1]))

        # initialize current param/spec observations
        self.cur_specs = np.zeros(len(self.specs_id), dtype=np.float32)
        self.cur_params_idx = np.zeros(len(self.params_id), dtype=np.int32)

        # Get the g* (overall design spec) you want to reach
        self.global_g = []
        for spec in list(self.specs.values()):
            self.global_g.append(float(spec[self.fixed_goal_idx]))
        self.g_star = np.array(self.global_g)
        self.global_g = np.array(yaml_data['normalize'])

        # objective number (used for validation)
        self.obj_idx = 0

    @debug_log
    def reset(self):
        # 合并多目标选择逻辑
        if self.generalize or self.multi_goal:
            if self.generalize and self.valid:
                # 顺序循环选择索引
                if self.obj_idx > self.num_os - 1:
                    self.obj_idx = 0
                idx = self.obj_idx
                self.obj_idx += 1
            else:
                # 随机选择索引
                idx = random.randint(0, self.num_os - 1)
            # 统一生成目标规格数组
            self.specs_ideal = np.array([spec[idx] for spec in self.specs.values()])
        else:
            # 单目标情况，使用固定规格
            self.specs_ideal = self.g_star

        # applicable only when you have multiple goals, normalizes everything to some global_g
        self.specs_ideal_norm = self.lookup(self.specs_ideal, self.global_g)

        # initialize current parameters
        self.cur_params_idx = np.array([33, 33, 33, 33, 33, 14, 20])
        self.cur_specs = self.update(self.cur_params_idx)
        cur_spec_norm = self.lookup(self.cur_specs, self.global_g)
        reward = self.reward(self.cur_specs, self.specs_ideal)

        # observation is a combination of current specs distance from ideal, ideal spec, and current param vals
        self.ob = np.concatenate([cur_spec_norm, self.specs_ideal_norm, self.cur_params_idx])
        return self.ob

    @debug_log
    def step(self, action):
        """
        :param action: is vector with elements between 0 and 1 mapped to the index of the corresponding parameter
        :return:
        """

        # Take action that RL agent returns to change current params
        action = list(np.reshape(np.array(action), (np.array(action).shape[0],)))
        self.cur_params_idx = self.cur_params_idx + np.array([self.action_meaning[a] for a in action])

        # self.cur_params_idx = self.cur_params_idx + np.array(self.action_arr[int(action)])
        self.cur_params_idx = np.clip(self.cur_params_idx, [0] * len(self.params_id),
                                      [(len(param_vec) - 1) for param_vec in self.params])
        # Get current specs and normalize
        self.cur_specs = self.update(self.cur_params_idx)
        cur_spec_norm = self.lookup(self.cur_specs, self.global_g)
        reward = self.reward(self.cur_specs, self.specs_ideal)
        done = reward >= 10  # 简化终止条件

        # Logging with single call if goal reached
        if done:
            log_details = (
                "\n{0}\n"
                "params = {1}\n"
                "specs: {2}\n"
                "ideal specs: {3}\n"
                "re: {4}\n"
                "{0}"
            ).format('-' * 10, self.cur_params_idx, self.cur_specs, self.specs_ideal, reward)

            log.info(log_details)  # 减少日志读写次数

        self.ob = np.concatenate([cur_spec_norm, self.specs_ideal_norm, self.cur_params_idx])
        self.env_steps += 1

        # print('cur ob:' + str(self.cur_specs))
        # print('ideal spec:' + str(self.specs_ideal))
        # print(reward)
        return self.ob, reward, done, {}

    @debug_log
    def lookup(self, spec, goal_spec):
        goal_spec = [float(e) for e in goal_spec]
        norm_spec = (spec - goal_spec) / (goal_spec + spec)
        return norm_spec

    @debug_log
    def reward(self, spec, goal_spec):
        '''
        Reward: doesn't penalize for overshooting spec, is negative
        '''
        rel_specs = self.lookup(spec, goal_spec)
        pos_val = []
        reward = 0.0
        for i, rel_spec in enumerate(rel_specs):
            if (self.specs_id[i] == 'ibias_max'):
                rel_spec = rel_spec * -1.0  # /10.0
            if rel_spec < 0:
                reward += rel_spec
                pos_val.append(0)
            else:
                pos_val.append(1)

        return reward if reward < -0.02 else 10

    @debug_log
    def update(self, params_idx):
        """
        :param action: an int between 0 ... n-1
        :return:
        """
        # impose constraint tail1 = in
        # params_idx[0] = params_idx[3]
        params = [self.params[i][params_idx[i]] for i in range(len(self.params_id))]
        param_val = [OrderedDict(list(zip(self.params_id, params)))]

        # run param vals and simulate
        cur_specs = OrderedDict(
            sorted(self.sim_env.create_design_and_simulate(param_val[0])[1].items(), key=lambda k: k[0]))
        cur_specs = np.array(list(cur_specs.values()))

        return cur_specs


def main():
    env_config = {"generalize": True, "valid": True}
    env = TwoStageAmp(env_config)
    env.reset()
    env.step([2, 2, 2, 2, 2, 2, 2])

    IPython.embed()


if __name__ == "__main__":
    main()
