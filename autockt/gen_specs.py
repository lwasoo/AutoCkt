import sys
from pathlib import Path

# 获取当前脚本的绝对路径（correct_inputs.py）
current_file = Path(__file__).resolve()
# 向上回溯到 AutoCkt 根目录（根据实际层级调整）
project_root = current_file.parent.parent  # 例如：../../..
sys.path.append(str(project_root))

from func_decorator import debug_log
import numpy as np
import random
import yaml
import os
import IPython
import argparse
from collections import OrderedDict
import pickle

from autockt.envs.read_yaml import OrderedDictYAMLLoader


# Generate the design specifications and then save to a pickle file
@debug_log
def gen_data(CIR_YAML, env, num_specs):
    with open(CIR_YAML, 'r') as f:
        yaml_data = yaml.load(f, OrderedDictYAMLLoader)

    specs_range = yaml_data['target_specs']  # 将param的spec固定在target_specs范围之内
    specs_range_vals = list(specs_range.values())  # 将specs_range转换为列表的形式
    specs_valid = []  # 生成空的列表
    for spec in specs_range_vals:  # 遍历yaml电路规格中的每一个specs
        if isinstance(spec[0], int):  # 若是int型使用random.randint()随机生成
            list_val = [random.randint(int(spec[0]), int(spec[1])) for x in range(0, num_specs)]
        else:  # 若是float型使用random.uniform()随机生成
            list_val = [random.uniform(float(spec[0]), float(spec[1])) for x in
                        range(0, num_specs)]  # 在理想规格中生成num_specs组数据
        specs_valid.append(tuple(list_val))  # 将list_val转为元组形式写到specs_valid列表当中
    i = 0
    for key, value in specs_range.items():  # 遍历原 YAML 文件中的 target_specs 规格
        specs_range[key] = specs_valid[i]  # 将 specs_valid 里的值按顺序赋回 specs_range
        i += 1
    with open("autockt/gen_specs/ngspice_specs_gen_" + env, 'wb') as f:
        pickle.dump(specs_range, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_specs', type=str)
    args = parser.parse_args()
    CIR_YAML = "eval_engines/ngspice/ngspice_inputs/yaml_files/two_stage_opamp.yaml"

    gen_data(CIR_YAML, "two_stage_opamp", int(args.num_specs))  # 由命令行规定的输入的num_specs


if __name__ == "__main__":
    main()
