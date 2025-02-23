from Log import log
from func_decorator import debug_log
import re
import numpy as np
import copy
from multiprocessing.dummy import Pool as ThreadPool
import os
import yaml
import random
import scipy.interpolate as interp
import scipy.optimize as sciopt
import time
import pprint
import IPython

debug = False


class SpectreWrapper(object):  # 修改: 类名从 NgSpiceWrapper 改为 SpectreWrapper
    BASE_TMP_DIR = os.path.abspath("/tmp/ckt_da")

    def __init__(self, num_process, yaml_path, path, root_dir=None):
        if root_dir is None:
            self.root_dir = SpectreWrapper.BASE_TMP_DIR  # 修改: 使用 SpectreWrapper 作为基类
        else:
            self.root_dir = root_dir

        with open(yaml_path, 'r') as f:
            yaml_data = yaml.load(f, Loader=yaml.FullLoader)
        design_netlist = yaml_data['dsn_netlist']
        design_netlist = path + '/' + design_netlist

        _, dsg_netlist_fname = os.path.split(design_netlist)
        self.base_design_name = os.path.splitext(dsg_netlist_fname)[0]
        self.num_process = num_process
        self.gen_dir = os.path.join(self.root_dir, "designs_" + self.base_design_name)

        os.makedirs(self.root_dir, exist_ok=True)
        os.makedirs(self.gen_dir, exist_ok=True)

        with open(design_netlist, 'r') as raw_file:
            self.tmp_lines = raw_file.readlines()

    @debug_log
    def get_design_name(self, state):
        fname = self.base_design_name
        for value in state.values():
            fname += "_" + str(value)
        return fname

    @debug_log
    def create_design(self, state, new_fname):
        design_folder = os.path.join(self.gen_dir, new_fname) + str(random.randint(0, 10000))
        os.makedirs(design_folder, exist_ok=True)

        fpath = os.path.join(design_folder, new_fname + '.scs')  # 修改: 生成的文件扩展名为 .scs

        lines = copy.deepcopy(self.tmp_lines)
        for line_num, line in enumerate(lines):
            if "include" in line:
              regex = re.compile("\include\s*\"(.*?)\"")
              found = regex.search(line)
              if found:
                pass  # Spectre 的 include 语句保持不变

            if ".parameter" in line:  # 修改: 适配 Spectre 语言中的参数设置
                for key, value in state.items():
                    regex = re.compile("%s=(\S+)" % (key))
                    found = regex.search(line)
                    if found:
                        new_replacement = "%s=%s" % (key, str(value))
                        lines[line_num] = lines[line_num].replace(found.group(0), new_replacement)


            if 'saveOptions' in line:  # 修改: Spectre 使用 saveOptions 进行数据存储
                regex = re.compile('save=allpub')
                if regex.search(line):
                    lines[line_num] = line.replace('save=allpub', 'save=all')

        with open(fpath, 'w') as f:
            f.writelines(lines)
        return design_folder, fpath


    @debug_log
    def simulate(self, fpath):
        info = 0  # 无错误时 info = 0
        command = "spectre %s > /dev/null 2>&1" % fpath  # 修改: 使用 Spectre 进行仿真
        exit_code = os.system(command)

        if debug:
            print(command)
            print(fpath)

        if exit_code % 256:
            info = 1  # 发生错误时 info = 1
        return info

    @debug_log
    def create_design_and_simulate(self, state, dsn_name=None, verbose=False):
        if debug:
            print('state', state)
            print('verbose', verbose)
        if dsn_name is None:
            dsn_name = self.get_design_name(state)
        else:
            dsn_name = str(dsn_name)
        if verbose:
            print(dsn_name)
        design_folder, fpath = self.create_design(state, dsn_name)
        info = self.simulate(fpath)
        specs = self.translate_result(design_folder)
        return state, specs, info

    @debug_log
    def run(self, states, design_names=None, verbose=False):
        pool = ThreadPool(processes=self.num_process)
        arg_list = [(state, dsn_name, verbose) for (state, dsn_name) in zip(states, design_names)]
        specs = pool.starmap(self.create_design_and_simulate, arg_list)
        pool.close()
        return specs

    @debug_log
    def translate_result(self, output_path):
        result = None  # 这里需要根据 Spectre 生成的仿真结果格式进行解析
        return result
