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


# way of ordering the way a yaml file is read
class OrderedDictYAMLLoader(yaml.Loader):
    """
    A YAML loader that loads mappings into ordered dictionaries.
    """

    def __init__(self, *args, **kwargs):
        yaml.Loader.__init__(self, *args, **kwargs)

        self.add_constructor(u'tag:yaml.org,2002:map', type(self).construct_yaml_map)
        self.add_constructor(u'tag:yaml.org,2002:omap', type(self).construct_yaml_map)

    @debug_log
    def construct_yaml_map(self, node):
        data = OrderedDict()
        yield data
        value = self.construct_mapping(node)
        data.update(value)

    @debug_log
    def construct_mapping(self, node, deep=False):
        if isinstance(node, yaml.MappingNode):  # isinstance的作用是检查某个对象是否是指定类型或类型的子类
            self.flatten_mapping(node)
        else:
            raise yaml.constructor.ConstructorError(None, None,
                                                    'expected a mapping node, but found %s' % node.id, node.start_mark)

        mapping = OrderedDict()
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)  # 使用construct_object对key_node进行映射
            value = self.construct_object(value_node, deep=deep)
            mapping[key] = value
        return mapping


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
    with open("autockt/gen_specs/spectre_specs_gen_" + env, 'wb') as f: #original ngspice_specs_gen_
        pickle.dump(specs_range, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_specs', type=str)
    args = parser.parse_args()
    CIR_YAML = "eval_engines/ngspice/ngspice_inputs/yaml_files/two_stage_opamp.yaml"

    gen_data(CIR_YAML, "two_stage_opamp", int(args.num_specs))  # 由命令行规定的输入的num_specs


if __name__ == "__main__":
    main()