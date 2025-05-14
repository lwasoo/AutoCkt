from Log import log
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
import shutil
import fasteners

debug = False


class NgSpiceWrapper(object):
    BASE_TMP_DIR = os.path.abspath(os.path.expanduser("~/workspace310a/ckt_da"))
    OCEAN_SCRIPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "export.ocn")

    def __init__(self, num_process, yaml_path, path, root_dir=None):
        if root_dir == None:
            self.root_dir = NgSpiceWrapper.BASE_TMP_DIR
        else:
            self.root_dir = root_dir

        with open(yaml_path, 'r') as f:
            yaml_data = yaml.load(f)
        design_netlist = yaml_data['dsn_netlist']
        design_netlist = path + '/' + design_netlist

        _, dsg_netlist_fname = os.path.split(design_netlist)
        self.base_design_name = os.path.splitext(dsg_netlist_fname)[0]
        self.num_process = num_process
        self.gen_dir = os.path.join(self.root_dir, "designs_" + self.base_design_name)
        self.simulation_count = 0  # 记录仿真次数
        self.cleanup_interval = 1000
        self.cleanup_lock = fasteners.InterProcessLock(os.path.join(self.gen_dir, "cleanup.lock"))  # 初始化锁文件

        # 创建目录，如果已存在不会抛出异常
        os.makedirs(self.root_dir, exist_ok=True)
        os.makedirs(self.gen_dir, exist_ok=True)

        raw_file = open(design_netlist, 'r')
        self.tmp_lines = raw_file.readlines()
        raw_file.close()

    def get_design_name(self, state):
        fname = self.base_design_name
        for value in state.values():
            fname += "_" + str(value)
        return fname

    def create_design(self, state, new_fname):
        """
         根据给定的参数 state 生成新的设计文件
         :param state: 电路设计参数字典
         :param new_fname: 生成的新文件名
         :return: 生成的电路设计文件所在的文件夹和路径
        """

        design_folder = os.path.join(self.gen_dir, new_fname) + str(random.randint(0, 10000))
        os.makedirs(design_folder, exist_ok=True)

        fpath = os.path.join(design_folder, new_fname + '.scs')

        lines = copy.deepcopy(self.tmp_lines)
        for line_num, line in enumerate(lines):
            if line.startswith("parameters"):
                for key, value in state.items():
                    regex = re.compile("(%s=\S+)" % key)  # 匹配 key=原值
                    found = regex.search(line)
                    if found:
                        new_replacement = "%s=%s" % (key, str(value))
                        line = line.replace(found.group(0), new_replacement)  # 替换所有匹配的值
                lines[line_num] = line  # 更新该行

        with open(fpath, 'w') as f:
            f.writelines(lines)
            f.close()

        # 读取 Ocean 脚本
        with open(NgSpiceWrapper.OCEAN_SCRIPT_PATH, "r") as f:
            tmp_lines = f.readlines()
        regex = re.compile(r'outfile\("([^"]+\.txt)"')  # 匹配 results.txt
        res_regex = re.compile(r'openResults\("([^"]+\.raw)"\)')
        for i, line in enumerate(tmp_lines):
            found = regex.search(line)
            if found:
                new_path = os.path.join(design_folder, "results.txt")  # 生成新路径
                new_line = line.replace(found.group(1), new_path)  # 替换文件名
                tmp_lines[i] = new_line

            res_found = res_regex.search(line)
            if res_found:
                new_path = os.path.join(design_folder, new_fname + '.raw')  # 生成新路径
                new_line = line.replace(res_found.group(1), new_path)  # 替换行中的文件名
                tmp_lines[i] = new_line

        ocean_script_path = os.path.join(design_folder, "export.ocn")  # 新的保存路径
        with open(ocean_script_path, "w") as f:
            f.writelines(tmp_lines)
        return design_folder, fpath

    def simulate(self, fpath):
        info = 0  # this means no error occurred
        # 获取fpath所在目录
        output_dir = os.path.dirname(fpath)
        if not os.path.isfile(os.path.join(output_dir, "export.ocn")):
            raise FileNotFoundError("Ocean script not found at expected path.")

        command = """
        spectre "{0}" -o "{1}" -log >& /dev/null
        ocean -nograph -restore "{2}" >& /dev/null
        find "{1}" -name ".*.dep" -exec rm -rf {{}} +
        find "{1}" -name "*.raw" -exec rm -rf {{}} +
        """.format(fpath, output_dir, os.path.join(output_dir, "export.ocn"))
        exit_code = os.system(command)
        if debug:
            print(command)
            print(fpath)

        if (exit_code % 256):
            info = 1  # this means an error has occurred

        return info

    def create_design_and_simulate(self, state, dsn_name=None, verbose=False):
        if debug:
            print('state', state)
            print('verbose', verbose)
        if dsn_name == None:
            dsn_name = self.get_design_name(state)
        else:
            dsn_name = str(dsn_name)
        if verbose:
            print(dsn_name)
        design_folder, fpath = self.create_design(state, dsn_name)
        info = self.simulate(fpath)
        specs = self.translate_result(design_folder)

        self.simulation_count += 1
        if self.simulation_count % self.cleanup_interval == 0:
            self.clean_old_simulations(keep_latest=100)

        return state, specs, info

    def clean_old_simulations(self, keep_latest=100):
        """
        删除较旧的仿真文件，只保留最近 keep_latest 次仿真结果。
        """
        # 增加进程锁
        if not self.cleanup_lock.acquire(blocking=False):
            return
        try:
            # 获取所有仿真生成的子目录
            design_folders = [
                d.path
                for d in os.scandir(self.gen_dir)
                if os.path.isdir(d.path)
            ]
            # 按创建时间排序（旧的在前，新的在后）
            # 缓存创建时间，减少重复调用 os.path.getctime
            design_folders = sorted(
                design_folders,
                key=lambda x: os.path.getctime(x)
            )

            if len(design_folders) > keep_latest:
                folders_to_delete = design_folders[:-keep_latest]  # 取出旧的部分
                for folder in folders_to_delete:
                    try:
                        shutil.rmtree(folder)  # 删除整个文件夹及其中所有内容
                    except Exception as e:
                        log.error("Error deleting folder {}: {}".format(folder, e))
        finally:
            self.cleanup_lock.release()

    def run(self, states, design_names=None, verbose=False):
        """

        :param states:
        :param design_names: if None default design name will be used, otherwise the given design name will be used
        :param verbose: If True it will print the design name that was created
        :return:
            results = [(state: dict(param_kwds, param_value), specs: dict(spec_kwds, spec_value), info: int)]
        """
        pool = ThreadPool(processes=self.num_process)  # 创建一个多线程池，进行多个仿真任务，self.num_process为线程数量
        arg_list = [(state, dsn_name, verbose) for (state, dsn_name) in zip(states, design_names)]
        specs = pool.starmap(self.create_design_and_simulate,
                             arg_list)  # starmap自动展开arg_list，传递参数给create_design_and_simulate进行设计仿真
        pool.close()
        return specs

    def translate_result(self, output_path):
        """
        This method needs to be overwritten according to cicuit needs,
        parsing output, playing with the results to get a cost function, etc.
        The designer should look at his/her netlist and accordingly write this function.

        :param output_path:
        :return:
        """
        result = None
        return result
