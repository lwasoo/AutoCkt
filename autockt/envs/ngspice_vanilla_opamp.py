"""
A new ckt environment based on a new structure of MDP
"""
import gym
from gym import spaces

import numpy as np
import random
import psutil

from multiprocessing.dummy import Pool as ThreadPool
from collections import OrderedDict
import yaml
import yaml.constructor
import statistics
import os
import IPython
import itertools
from eval_engines.util.core import *
import pickle
import os

from eval_engines.ngspice.TwoStageClass import *

# 自定义 YAML 加载器，确保加载的字典保持顺序
class OrderedDictYAMLLoader(yaml.Loader):
    """
    A YAML loader that loads mappings into ordered dictionaries.
    """

    def __init__(self, *args, **kwargs):
        yaml.Loader.__init__(self, *args, **kwargs)

        # 添加构造器，确保 YAML 中的映射和有序映射都加载为 OrderedDict
        self.add_constructor(u'tag:yaml.org,2002:map', type(self).construct_yaml_map)
        self.add_constructor(u'tag:yaml.org,2002:omap', type(self).construct_yaml_map)

    def construct_yaml_map(self, node):
        data = OrderedDict()
        yield data
        value = self.construct_mapping(node)
        data.update(value)

    def construct_mapping(self, node, deep=False):
        if isinstance(node, yaml.MappingNode):
            self.flatten_mapping(node)
        else:
            raise yaml.constructor.ConstructorError(None, None,
                                                    'expected a mapping node, but found %s' % node.id, node.start_mark)

        mapping = OrderedDict()
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            value = self.construct_object(value_node, deep=deep)
            mapping[key] = value
        return mapping

# 定义两级运算放大器的强化学习环境
class TwoStageAmp(gym.Env):
    metadata = {'render.modes': ['human']}  # 环境的元数据，定义渲染模式,定义人类可读的渲染模式

    PERF_LOW = -1  # 性能指标的下限
    PERF_HIGH = 0  # 性能指标的上限

    # 获取 YAML 文件的路径
    path = os.getcwd() #获取当前工作目录
    CIR_YAML = path + '/eval_engines/ngspice/ngspice_inputs/yaml_files/two_stage_opamp.yaml'

def __init__(self, env_config): #self是类的实例的引用，通过它可以访问实例的属性和方法
        # 从环境配置中获取参数
        self.multi_goal = env_config.get("multi_goal", False)  # 是否使用多目标优化，查找multi_goal的值，若无法再env.config的字典中查找到multi_goal的值，则返回False
        self.generalize = env_config.get("generalize", False)  # 是否使用广义规格
        num_valid = env_config.get("num_valid", 50)  # 验证集的数量
        self.specs_save = env_config.get("save_specs", False)  # 是否保存规格
        self.valid = env_config.get("run_valid", False)  # 是否运行验证

        self.env_steps = 0  # 环境步数计数器

        # 加载 YAML 文件中的电路设计参数和目标规格,使用的是OrderedDictYAMLLoader类进行加载
        with open(TwoStageAmp.CIR_YAML, 'r') as f:
            yaml_data = yaml.load(f, OrderedDictYAMLLoader)

        # 加载设计规格
        if self.generalize == False:
            specs = yaml_data['target_specs']  # 从 YAML 文件中加载目标规格
        else:
            # 从预先生成的规格文件中加载规格
            load_specs_path = TwoStageAmp.path + "/autockt/gen_specs/ngspice_specs_gen_two_stage_opamp"
            with open(load_specs_path, 'rb') as f:
                specs = pickle.load(f) #使用 pickle 模块从文件对象 f 中加载序列化的数据，并将其反序列化为 Python 对象。

        # 对规格进行排序并存储
        self.specs = OrderedDict(sorted(specs.items(), key=lambda k: k[0]))
        if self.specs_save:
            # 如果需要保存规格，则将规格保存到文件中
            with open("specs_" + str(num_valid) + str(random.randint(1, 100000)), 'wb') as f:
                pickle.dump(self.specs, f) #转化为二进制格式方便保存

        self.specs_ideal = []  # 存储理想规格
        self.specs_id = list(self.specs.keys())  # 存储规格的名称
        self.fixed_goal_idx = -1  # 固定目标规格的索引
        self.num_os = len(list(self.specs.values())[0])  # 规格的数量

        # 加载参数
        params = yaml_data['params']
        self.params = []  # 存储每个参数的取值范围
        self.params_id = list(params.keys())  # 存储参数的名称

        for value in params.values():
            # 生成参数的取值范围
            param_vec = np.arange(value[0], value[1], value[2])
            self.params.append(param_vec)

        # 初始化仿真环境
        self.sim_env = TwoStageClass(yaml_path=TwoStageAmp.CIR_YAML, num_process=1, path=TwoStageAmp.path)
        self.action_meaning = [-1, 0, 2]  # 定义动作的含义：-1 表示减小，0 表示不变，2 表示增加
        self.action_space = spaces.Tuple([spaces.Discrete(len(self.action_meaning))] * len(self.params_id))  # 定义动作空间
        self.observation_space = spaces.Box(
            low=np.array([TwoStageAmp.PERF_LOW] * 2 * len(self.specs_id) + len(self.params_id) * [1]),
            high=np.array([TwoStageAmp.PERF_HIGH] * 2 * len(self.specs_id) + len(self.params_id) * [1]))  # 定义观察空间
        #前几个维度用于性能指标，后几个维度用于参数的id名称

        # 初始化当前参数和规格
        self.cur_specs = np.zeros(len(self.specs_id), dtype=np.float32)  # 当前规格
        self.cur_params_idx = np.zeros(len(self.params_id), dtype=np.int32)  # 当前参数索引

        # 获取全局目标规格
        self.global_g = []
        for spec in list(self.specs.values()):
            self.global_g.append(float(spec[self.fixed_goal_idx])) #将值转化为浮点数并添加到global_g的全局变量当中
        self.g_star = np.array(self.global_g)  # 全局目标规格
        self.global_g = np.array(yaml_data['normalize'])  # 规格的归一化值

        # 目标规格的索引（用于验证）
        self.obj_idx = 0

    def reset(self):
        # 如果使用广义规格，则在每次重置时选择不同的目标规格
        if self.generalize == True:
            if self.valid == True:
                if self.obj_idx > self.num_os - 1: #self.obj_idx为索引，跟踪目标规格的数量
                    self.obj_idx = 0
                idx = self.obj_idx
                self.obj_idx += 1
            else:
                idx = random.randint(0, self.num_os - 1)  #随机选择0至self.num_os-1的值
            self.specs_ideal = []
            for spec in list(self.specs.values()):
                self.specs_ideal.append(spec[idx])  #添加spec[idx]值到specs_ideal变量当中
            self.specs_ideal = np.array(self.specs_ideal)
        else:
            if self.multi_goal == False:
                self.specs_ideal = self.g_star  # 使用全局目标规格
            else:
                idx = random.randint(0, self.num_os - 1)
                self.specs_ideal = []
                for spec in list(self.specs.values()):
                    self.specs_ideal.append(spec[idx])
                self.specs_ideal = np.array(self.specs_ideal)

        # 对目标规格进行归一化
        self.specs_ideal_norm = self.lookup(self.specs_ideal, self.global_g)

        # 初始化当前参数
        self.cur_params_idx = np.array([33, 33, 33, 33, 33, 14, 20])
        self.cur_specs = self.update(self.cur_params_idx)  # 更新当前规格
        cur_spec_norm = self.lookup(self.cur_specs, self.global_g)  # 对当前规格进行归一化，lookup函数用于计算归一化的差距
        reward = self.reward(self.cur_specs, self.specs_ideal)  # 计算奖励

        # 观察值是当前规格与目标规格的差异、目标规格和当前参数的组合
        self.ob = np.concatenate([cur_spec_norm, self.specs_ideal_norm, self.cur_params_idx])
        return self.ob

    def step(self, action):

        #:param action: 动作向量，元素为 0、1 或 2，表示对每个参数的操作
        #:return: 新的观察值、奖励、是否结束、额外信息
        

        # 执行动作，调整当前参数
        action = list(np.reshape(np.array(action), (np.array(action).shape[0],))) #将action重塑为一个一维的数组，并将重塑的一维数组通过list转换为python列表
        self.cur_params_idx = self.cur_params_idx + np.array([self.action_meaning[a] for a in action]) #这是一个列表推导式，遍历 action 中的每个动作 a 

        # 确保参数索引在合法范围内
        self.cur_params_idx = np.clip(self.cur_params_idx, #需要裁剪的数组
                                      [0] * len(self.params_id), #最小值
                                      [(len(param_vec) - 1) for param_vec in self.params]) #最大值
        # 更新当前规格并归一化
        self.cur_specs = self.update(self.cur_params_idx)
        cur_spec_norm = self.lookup(self.cur_specs, self.global_g)
        reward = self.reward(self.cur_specs, self.specs_ideal)  # 计算奖励
        done = False

        # 如果奖励大于等于 10，表示达到目标状态，结束当前回合
        if (reward >= 10):
            done = True
            print('-' * 10)
            print('params = ', self.cur_params_idx)
            print('specs:', self.cur_specs)
            print('ideal specs:', self.specs_ideal)
            print('re:', reward)
            print('-' * 10)

        # 更新观察值
        self.ob = np.concatenate([cur_spec_norm, self.specs_ideal_norm, self.cur_params_idx])
        self.env_steps = self.env_steps + 1

        return self.ob, reward, done, {}

    def lookup(self, spec, goal_spec):
        
        #计算当前规格与目标规格的归一化差异
        
        goal_spec = [float(e) for e in goal_spec]
        norm_spec = (spec - goal_spec) / (goal_spec + spec)
        return norm_spec

    def reward(self, spec, goal_spec):
        
        #计算奖励，奖励为负值，表示当前规格与目标规格的差异
        
        rel_specs = self.lookup(spec, goal_spec)
        pos_val = []
        reward = 0.0
        for i, rel_spec in enumerate(rel_specs):
            if (self.specs_id[i] == 'ibias_max'):
                rel_spec = rel_spec * -1.0  # 对于 ibias_max，反转差异
            if rel_spec < 0:
                reward += rel_spec  # 累加负差异
                pos_val.append(0)
            else:
                pos_val.append(1)

        return reward if reward < -0.02 else 10  # 如果差异很小，返回 10 表示达到目标

    def update(self, params_idx):
        
        #根据参数索引更新电路设计，并返回当前规格
        
        params = [self.params[i][params_idx[i]] for i in range(len(self.params_id))] #从二维数组中提取参数值
        param_val = [OrderedDict(list(zip(self.params_id, params)))] #构造键对值，并使之通过OrdereDict构造列表

        # 运行仿真并获取当前规格
        cur_specs = OrderedDict(sorted(self.sim_env.create_design_and_simulate(param_val[0])[1].items(), key=lambda k: k[0]))
        cur_specs = np.array(list(cur_specs.values()))

        return cur_specs

# 主函数，用于测试环境
def main():
    env_config = {"generalize": True, "valid": True}  # 环境配置
    env = TwoStageAmp(env_config)  # 创建环境实例
    env.reset()  # 重置环境
    env.step([2, 2, 2, 2, 2, 2, 2])  # 执行一个动作

    IPython.embed()  # 启动交互式调试

# 如果当前脚本作为主程序运行，则执行 main 函数
if __name__ == "__main__":
    main()
