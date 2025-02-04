import re
import numpy as np
import copy
from multiprocessing.dummy import Pool as ThreadPool
import os
import abc
import scipy.interpolate as interp
import scipy.optimize as sciopt
import random
import time
import pprint
import yaml
import IPython

# 调试模式开关
debug = False

class NgSpiceWrapper(object):
    # NGSpice 仿真器封装类，用于管理电路仿真、参数配置、结果解析等

    BASE_TMP_DIR = os.path.abspath("/tmp/ckt_da")  # 设定默认的临时目录

    def __init__(self, num_process, yaml_path, path, root_dir=None):
        # 初始化 NGSpice 仿真器封装
        # :param num_process: 并行进程数
        # :param yaml_path: YAML 配置文件路径
        # :param path: 设计文件路径
        # :param root_dir: 根目录（可选），默认为 BASE_TMP_DIR
        
        if root_dir is None:
            self.root_dir = NgSpiceWrapper.BASE_TMP_DIR
        else:
            self.root_dir = root_dir

        # 读取 YAML 文件
        with open(yaml_path, 'r') as f:
            yaml_data = yaml.load(f, Loader=yaml.SafeLoader)
        
        design_netlist = yaml_data['dsn_netlist']
        design_netlist = path + '/' + design_netlist

        _, dsg_netlist_fname = os.path.split(design_netlist)
        self.base_design_name = os.path.splitext(dsg_netlist_fname)[0]
        self.num_process = num_process
        self.gen_dir = os.path.join(self.root_dir, "designs_" + self.base_design_name)

        # 确保目录存在
        os.makedirs(self.root_dir, exist_ok=True)
        os.makedirs(self.gen_dir, exist_ok=True)

        # 读取原始 netlist 文件
        with open(design_netlist, 'r') as raw_file:
            self.tmp_lines = raw_file.readlines()

    def get_design_name(self, state):
        # 生成唯一的设计名称，包含所有状态变量值
        fname = self.base_design_name
        for value in state.values():
            fname += "_" + str(value)
        return fname

    def create_design(self, state, new_fname):
        # 根据给定的参数 state 生成新的设计文件
        # :param state: 电路设计参数字典
        # :param new_fname: 生成的新文件名
        # :return: 生成的电路设计文件所在的文件夹和路径
        
        design_folder = os.path.join(self.gen_dir, new_fname) + str(random.randint(0, 10000)) #生成文件夹的唯一路径
        os.makedirs(design_folder, exist_ok=True)

        fpath = os.path.join(design_folder, new_fname + '.cir') #生成文件的唯一路径

        lines = copy.deepcopy(self.tmp_lines) 
        for line_num, line in enumerate(lines): #遍历lines的每一行并获取行号和行的内容
            # 处理 `.include` 语句
            if '.include' in line:
                regex = re.compile(r"\.include\s*\"(.*?)\"") 
                found = regex.search(line)
                if found:
                    pass  # 不修改模型路径，保留原始 include 语句

            # 处理 `.param` 语句，替换参数值，提取参数值
            if '.param' in line:
                for key, value in state.items():
                    regex = re.compile(r"%s=(\S+)" % key) #用于提取param值，显示为例如W=u的形式
                    found = regex.search(line)      
                    if found:
                        new_replacement = "%s=%s" % (key, str(value))#遍历每一个字典，查找有无匹配的key=value的形式的param，若有，则需要用regex函数进行匹配并进行提取
                        lines[line_num] = lines[line_num].replace(found.group(0), new_replacement) #使用replace函数进行替换

            # 处理 `wrdata` 语句，确保输出文件路径正确，使用NGSpice的存储路径，将数据写入文件
            if 'wrdata' in line:
                regex = re.compile(r"wrdata\s*(\w+\.\w+)\s*")#捕获文件名，要求文件名格式为"xxx.yyy"（使用（\w+\.\w+)捕获文件名），前后是消除空格影响的语法
                found = regex.search(line)
                if found:
                    replacement = os.path.join(design_folder, found.group(1)) #合并路径
                    lines[line_num] = lines[line_num].replace(found.group(1), replacement) #使用replace函数进行替换

        # 写入新的 netlist 文件
        with open(fpath, 'w') as f:
            f.writelines(lines)
        return design_folder, fpath

    def simulate(self, fpath):
        # 运行 NGSpice 进行仿真
        # :param fpath: 生成的 netlist 文件路径
        # :return: info = 0 表示无错误，info = 1 表示出现错误
        
        info = 0  # 默认无错误
        command = "ngspice -b %s >/dev/null 2>&1" % fpath #NGSpice的命令形式，目的是运行NGSpice仿真并且防止NGSpice有过多的输出
        exit_code = os.system(command) #os.system返回command的输出状态，值为非0和0两种

        if debug:
            print(command)
            print(fpath)

        if exit_code % 256:
            info = 1  # 发生错误
        return info

    def create_design_and_simulate(self, state, dsn_name=None, verbose=False):
        # 创建电路设计并运行仿真，目的是整合以上define的函数，并使用它们，将他们定义成整体的函数，以供设计和仿真
        # :param state: 设计参数
        # :param dsn_name: 设计名称（可选）
        # :param verbose: 是否打印详细信息
        # :return: (state, specs, info)，即设计参数、仿真结果、仿真状态
        
        if debug:
            print('state', state)
            print('verbose', verbose)

        if dsn_name is None:
            dsn_name = self.get_design_name(state)
        else:
            dsn_name = str(dsn_name)

        if verbose:
            print(dsn_name)

        design_folder, fpath = self.create_design(state, dsn_name) #使用create_design函数进行输出
        info = self.simulate(fpath) #使用simulate函数进行输出
        specs = self.translate_result(design_folder)
        return state, specs, info

    def run(self, states, design_names=None, verbose=False):
        # 并行运行多个电路设计仿真任务
        # :param states: 设计参数列表
        # :param design_names: 设计名称列表（可选）
        # :param verbose: 是否打印详细信息
        # :return: [(state, specs, info)]，即每个设计的参数、仿真结果、仿真状态
        
        pool = ThreadPool(processes=self.num_process) #创建一个多线程池，进行多个仿真任务，self.num_process为线程数量
        arg_list = [(state, dsn_name, verbose) for (state, dsn_name) in zip(states, design_names)]
        specs = pool.starmap(self.create_design_and_simulate, arg_list) #starmap自动展开arg_;ist，传递参数给create_design_and_simulate进行设计仿真
        pool.close()
        return specs

    def translate_result(self, output_path):
        # 解析仿真结果（需要根据具体电路进行重写）
        # :param output_path: 仿真输出文件夹
        # :return: 解析后的结果字典（默认为 None，需要用户自定义）
        
        result = None
        return result
