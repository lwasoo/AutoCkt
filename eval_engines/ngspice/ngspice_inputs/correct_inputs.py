import sys
from pathlib import Path

# 获取当前脚本的绝对路径（correct_inputs.py）
current_file = Path(__file__).resolve()
# 向上回溯到 AutoCkt 根目录（根据实际层级调整）
project_root = current_file.parent.parent.parent.parent  # 例如：../../..
sys.path.append(str(project_root))


#该文件主要用于这段代码用于自动更新 SPICE 电路仿真文件（.cir 文件）中 .include 语句的路径，
#确保它们正确指向 spice_models/45nm_bulk.txt，避免因为路径错误导致 SPICE 仿真失败。


from Log import log
from func_decorator import debug_log
import os
import re

@debug_log
def update_file(fname, path_to_model):
    log.info("changing {}".format(fname)) #log.info函数便是用记录日志打印fname
    with open(fname, 'r') as f:
        lines = f.readlines()

    for line_num, line in enumerate(lines):
        if '.include' in line:
            regex = re.compile("\.include\s*\"(.*?45nm\_bulk\.txt)\"")
            found = regex.search(line)
            if found:
                lines[line_num] = lines[line_num].replace(found.group(1), path_to_model)

    with open(fname, 'w') as f:
        f.writelines(lines)
        f.close()

if __name__ == '__main__':
    cur_fpath = os.path.realpath(__file__)
    parent_path = os.path.abspath(os.path.join(cur_fpath, os.pardir))
    netlist_path = os.path.join(parent_path, 'netlist')
    spice_model = os.path.join(parent_path, 'spice_models/45nm_bulk.txt')

    for root, dirs, files in os.walk(netlist_path):
        for f in files:
            if f.endswith(".cir"):
                update_file(fname=os.path.join(root, f), path_to_model=spice_model)