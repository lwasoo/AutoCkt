import numpy as np
import os
import scipy.interpolate as interp
import scipy.optimize as sciopt
import yaml
import importlib
import time

# 调试模式开关
debug = False

# 导入 NGSpice 仿真接口
from eval_engines.ngspice.ngspice_wrapper import NgSpiceWrapper

class TwoStageClass(NgSpiceWrapper):
    # 该类用于处理二级运算放大器的仿真结果，并计算关键指标。
    # 继承自 NgSpiceWrapper，提供仿真输出解析和性能计算方法。

    def translate_result(self, output_path):
        # 解析仿真输出文件，并计算关键性能指标。
        # :param output_path: 仿真输出目录
        # :return: spec (dict)，包含增益、单位增益带宽 (UGBW)、相位裕量 (PHM)、偏置电流 (Ibias)
        freq, vout, ibias = self.parse_output(output_path)  # 解析仿真输出
        gain = self.find_dc_gain(vout)  # 计算 DC 增益
        ugbw = self.find_ugbw(freq, vout)  # 计算单位增益带宽
        phm = self.find_phm(freq, vout)  # 计算相位裕量

        spec = {
            "ugbw": ugbw,
            "gain": gain,
            "phm": phm,
            "ibias": ibias
        }
        return spec

    def parse_output(self, output_path):
        # 解析 AC 和 DC 仿真结果文件。
        # :param output_path: 仿真结果目录
        # :return: freq (频率数据), vout (输出电压数据), ibias (偏置电流)
        ac_fname = os.path.join(output_path, 'ac.csv')
        dc_fname = os.path.join(output_path, 'dc.csv')

        if not os.path.isfile(ac_fname) or not os.path.isfile(dc_fname):
            print("ac/dc file doesn't exist: %s" % output_path)

        ac_raw_outputs = np.genfromtxt(ac_fname, skip_header=1)
        dc_raw_outputs = np.genfromtxt(dc_fname, skip_header=1)
        freq = ac_raw_outputs[:, 0]
        vout_real = ac_raw_outputs[:, 1]
        vout_imag = ac_raw_outputs[:, 2]
        vout = vout_real + 1j * vout_imag  # 复数形式表示 AC 响应
        ibias = -dc_raw_outputs[1]  # 提取偏置电流

        return freq, vout, ibias

    def find_dc_gain(self, vout):
        # 计算 DC 增益
        return np.abs(vout)[0]

    def find_ugbw(self, freq, vout):
        # 计算单位增益带宽 (UGBW)
        gain = np.abs(vout)
        ugbw, valid = self._get_best_crossing(freq, gain, val=1)
        return ugbw if valid else freq[0]

    def find_phm(self, freq, vout):
        # 计算相位裕量 (Phase Margin, PHM)
        gain = np.abs(vout)
        phase = np.angle(vout, deg=False)
        phase = np.unwrap(phase)  # 处理相位跳变
        phase = np.rad2deg(phase)  # 转换为角度

        phase_fun = interp.interp1d(freq, phase, kind='quadratic')
        ugbw, valid = self._get_best_crossing(freq, gain, val=1)
        if valid:
            return -180 + phase_fun(ugbw) if phase_fun(ugbw) > 0 else 180 + phase_fun(ugbw)
        else:
            return -180

    def _get_best_crossing(cls, xvec, yvec, val):
        # 计算 yvec 在 xvec 轴上最接近 val 的交点
        interp_fun = interp.InterpolatedUnivariateSpline(xvec, yvec)

        def fzero(x):
            return interp_fun(x) - val

        xstart, xstop = xvec[0], xvec[-1]
        try:
            return sciopt.brentq(fzero, xstart, xstop), True
        except ValueError:
            return xstop, False

class TwoStageMeasManager:
    # 该类管理仿真测量过程，加载设计规格并计算优化成本。

    def __init__(self, design_specs_fname):
        # 初始化测量管理器，读取设计规格文件。
        self.design_specs_fname = design_specs_fname
        with open(design_specs_fname, 'r') as f:
            self.ver_specs = yaml.safe_load(f)

        self.spec_range = self.ver_specs['spec_range']
        self.params = self.ver_specs['params']

    def evaluate(self, design):
        # 评估设计参数并计算成本。
        state_dict = {key: self.params[key][design[i]] for i, key in enumerate(self.params.keys())}
        state = [state_dict]
        results = {netlist_name: netlist_module.run(state, [design.id]) for netlist_name, netlist_module in self.netlist_module_dict.items()}
        specs_dict = self._get_specs(results)
        specs_dict['cost'] = self.cost_fun(specs_dict)
        return specs_dict

    def _get_specs(self, results_dict):
        # 解析仿真结果，计算关键指标。
        ugbw_cur = results_dict['ol'][0][1]['ugbw']
        gain_cur = results_dict['ol'][0][1]['gain']
        phm_cur = results_dict['ol'][0][1]['phm']
        ibias_cur = results_dict['ol'][0][1]['Ibias']
        specs_dict = {
            "gain": gain_cur,
            "ugbw": ugbw_cur,
            "pm": phm_cur,
            "ibias": ibias_cur
        }
        return specs_dict

    def cost_fun(self, specs_dict):
        # 计算优化成本函数。
        return sum(self.compute_penalty(specs_dict[spec], spec)[0] for spec in self.spec_range.keys())
