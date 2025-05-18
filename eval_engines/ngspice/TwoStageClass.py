from Log import log

import numpy as np
import os
import scipy.interpolate as interp
import scipy.optimize as sciopt
import yaml
import importlib
import time

debug = False

from eval_engines.ngspice.ngspice_wrapper import NgSpiceWrapper


class TwoStageClass(NgSpiceWrapper):

    def translate_result(self, output_path):
        """
        :param output_path:
        :return
            result: dict(spec_kwds, spec_value)
        """

        result_file = os.path.join(output_path, 'results.txt')
    
        if not os.path.isfile(result_file):
            log.warning("results file doesn't exist: {}".format(output_path))
            return None

        key_map = {
            'VoltageGain': 'gain',
            'ugbw': 'ugbw',
            'phm': 'phm',
            'ibias': 'ibias'
        }

        results = {}

        with open(result_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    key, value = line.split(':')
                    key = key.strip()
                    value = float(value.strip())
                    mapped_key = key_map.get(key)
                    if mapped_key:
                        results[mapped_key] = -value if mapped_key == 'ibias' else value
                except Exception as e:
                    log.warning("[translate_result] Skipping line due to error: {} ({})".format(line, e))
                    continue

        missing = [k for k in ['ugbw', 'gain', 'phm', 'ibias'] if results.get(k) is None]
        if missing:
            log.warning("[translate_result] Missing spec(s) in {}: {}".format(output_path, ", ".join(missing)))

        return dict(
            ugbw=results.get('ugbw') if results.get('ugbw') is not None else 1e3,
            gain=results.get('gain') if results.get('gain') is not None else 0.001,
            phm=results.get('phm') if results.get('phm') is not None else 0.0,
            ibias=results.get('ibias') if results.get('ibias') is not None else 1e-6
        )

#原作者遗留，目前为止没用过
class TwoStageMeasManager(object):

    def __init__(self, design_specs_fname):
        self.design_specs_fname = design_specs_fname
        with open(design_specs_fname, 'r') as f:
            self.ver_specs = yaml.load(f)

        self.spec_range = self.ver_specs['spec_range']
        self.params = self.ver_specs['params']

        self.params_vec = {}
        self.search_space_size = 1
        for key, value in self.params.items():
            if value is not None:
                # self.params_vec contains keys of the main parameters and the corresponding search vector for each
                self.params_vec[key] = np.arange(value[0], value[1], value[2]).tolist()
                self.search_space_size = self.search_space_size * len(self.params_vec[key])

        self.measurement_specs = self.ver_specs['measurement']
        root_dir = self.measurement_specs['root_dir'] + "_" + time.strftime("%d-%m-%Y_%H-%M-%S")
        num_process = self.measurement_specs['num_process']

        self.netlist_module_dict = {}
        for netlist_kwrd, netlist_val in self.measurement_specs['netlists'].items():
            netlist_module = importlib.import_module(netlist_val['wrapper_module'])
            netlist_cls = getattr(netlist_module, netlist_val['wrapper_class'])
            self.netlist_module_dict[netlist_kwrd] = netlist_cls(num_process=num_process,
                                                                 design_netlist=netlist_val['cir_path'],
                                                                 root_dir=root_dir)

    def evaluate(self, design):
        state_dict = dict()
        for i, key in enumerate(self.params_vec.keys()):
            state_dict[key] = self.params_vec[key][design[i]]
        state = [state_dict]
        dsn_names = [design.id]
        results = {}
        for netlist_name, netlist_module in self.netlist_module_dict.items():
            results[netlist_name] = netlist_module.run(state, dsn_names)

        specs_dict = self._get_specs(results)
        specs_dict['cost'] = self.cost_fun(specs_dict)
        return specs_dict

    def _get_specs(self, results_dict):
        fdbck = self.measurement_specs['tb_params']['feedback_factor']
        tot_err = self.measurement_specs['tb_params']['tot_err']

        ugbw_cur = results_dict['ol'][0][1]['ugbw']
        gain_cur = results_dict['ol'][0][1]['gain']
        phm_cur = results_dict['ol'][0][1]['phm']
        ibias_cur = results_dict['ol'][0][1]['Ibias']

        # common mode gain and cmrr
        cm_gain_cur = results_dict['cm'][0][1]['cm_gain']
        cmrr_cur = 20 * np.log10(gain_cur / cm_gain_cur)  # in db
        # power supply gain and psrr
        ps_gain_cur = results_dict['ps'][0][1]['ps_gain']
        psrr_cur = 20 * np.log10(gain_cur / ps_gain_cur)  # in db

        # transient settling time and offset calculation
        t = results_dict['tran'][0][1]['time']
        vout = results_dict['tran'][0][1]['vout']
        vin = results_dict['tran'][0][1]['vin']

        tset_cur = self.netlist_module_dict['tran'].get_tset(t, vout, vin, fdbck, tot_err=tot_err)
        offset_curr = abs(vout[0] - vin[0] / fdbck)

        specs_dict = dict(
            gain=gain_cur,
            ugbw=ugbw_cur,
            pm=phm_cur,
            ibias=ibias_cur,
            cmrr=cmrr_cur,
            psrr=psrr_cur,
            offset_sys=offset_curr,
            tset=tset_cur,
        )

        return specs_dict

    def compute_penalty(self, spec_nums, spec_kwrd):
        if type(spec_nums) is not list:
            spec_nums = [spec_nums]
        penalties = []
        for spec_num in spec_nums:
            penalty = 0
            spec_min, spec_max, w = self.spec_range[spec_kwrd]
            if spec_max is not None:
                if spec_num > spec_max:
                    # penalty += w*abs((spec_num - spec_max) / (spec_num + spec_max))
                    penalty += w * abs(spec_num - spec_max) / abs(spec_num)
            if spec_min is not None:
                if spec_num < spec_min:
                    # penalty += w*abs((spec_num - spec_min) / (spec_num + spec_min))
                    penalty += w * abs(spec_num - spec_min) / abs(spec_min)
            penalties.append(penalty)
        return penalties

    def cost_fun(self, specs_dict):
        """
        :param design: a list containing relative indices according to yaml file
        :param verbose:
        :return:
        """
        cost = 0
        for spec in self.spec_range.keys():
            penalty = self.compute_penalty(specs_dict[spec], spec)[0]
            cost += penalty

        return cost
