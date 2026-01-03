import pandas as pd
import numpy as np
import csv
import pickle
from copy import deepcopy

def log_uniform_10(lower, upper, size=None):
    """
    Used for log-uniform sampling.
    :param lower: lower bound (> 0)
    :param upper: upper bound (> 0)
    :param size: number of samples
    :return:
    """
    return np.power(10.0, np.random.uniform(np.log10(lower), np.log10(upper), size=size))


# used to indicate when a parameter is completely deactivated
class NA:
    def __repr__(self):
        return 'n/a'

    def __str__(self):
        return 'n/a'

# TODO: have parameter_default be optional and adjust label assignments for this
class ParameterSpace:
    def __init__(self, filepath: str, parameter_domain: dict, parameter_type: dict, parameter_default: dict):
        self.filepath = filepath
        self.parameter_domain = parameter_domain
        self.parameter_type = parameter_type
        self.parameter_default = parameter_default
        self.parameter_design = None
        self.parameter_data = None

    def to_acts(self):
        """
        save parameter space as ACTS string, for use in covering array solver
        :return:
        """
        acts_string = '[System]\nName: parameter_space\n\n[Parameter]\n'
        for parameter in self.parameter_domain.keys():
            acts_string += parameter + '(enum): '
            num_values = len(self.parameter_domain[parameter])
            for value in range(num_values):
                acts_string += str(value)
                if value < num_values - 1:
                    acts_string += ','
            acts_string += '\n'
        acts_string += '\n[Constraint]\n'

        # save parameter space
        with open(self.filepath + 'parameter_domain.pkl', 'wb') as file:
            pickle.dump(self.parameter_domain, file)

        with open(self.filepath + 'parameter_type.pkl', 'wb') as file:
            pickle.dump(self.parameter_type, file)

        with open(self.filepath + 'parameter_default.pkl', 'wb') as file:
            pickle.dump(self.parameter_default, file)

        with open(self.filepath + 'parameter_space.txt', 'w') as file:
            file.write(acts_string)

        print('Saved ' + self.filepath + 'parameter_space.txt' +
              '\nThis can be imported at https://srd.sba-research.org/tools/cagen/ to generate a covering array. Export to .csv and save the file in the same directory as parameter_space.txt. Remember to include any constraints or conditionality in the Constraints field of CAGen.')

        return acts_string

    def get_parameter_design(self, filename: str, unfilled: str = '*'):
        with open(filename, 'r') as file:
            parameter_design = list(csv.DictReader(file))
        # TODO: what about conditional parameters?
        for config in parameter_design:
            for parameter in self.parameter_domain.keys():
                if config[parameter] == unfilled:
                    assert False, "Use 'Randomize Don't-Care Values' in CAGen."
                    # FIXME: disabled since randomization is done through CAGen
                    # # config[parameter] = np.random.choice(self.parameter_domain[parameter])
                    # val_idx = np.random.randint(len(self.parameter_domain[parameter]))
                    # config[parameter] = self.parameter_domain[parameter][val_idx]
                    # # TODO: take constraints into account more directly
                    # while isinstance(config[parameter], NA):
                    #     val_idx = np.random.randint(len(self.parameter_domain[parameter]))
                    #     config[parameter] = self.parameter_domain[parameter][val_idx]
                else:
                    config[parameter] = self.parameter_domain[parameter][int(config[parameter])]
        if self.parameter_default not in parameter_design:
            parameter_design = [self.parameter_default] + parameter_design
        with open(self.filepath + 'parameter_design.pkl', 'wb') as file:
            pickle.dump(parameter_design, file)
        self.parameter_design = parameter_design
        return parameter_design

    def get_parameter_data(self, parameter_design: list, use_log_uniform: list = None):
        """
        Get the parameter data, filling in intervals with sampled values from within those intervals.
        :param parameter_domain:
        :param parameter_design:
        :param use_log_uniform: list of parameters to use log10-uniform sampling in interval, else uniform
        :return:
        """
        # FIXME: Assumes positive lower/upper ranges when using log
        # get the parameter levels that are intervals
        interval_levels = dict()
        for parameter in self.parameter_domain.keys():
            interval_levels[parameter] = []
            for level in self.parameter_domain[parameter]:
                if isinstance(level, tuple):
                    interval_levels[parameter].append(level)
        # for each interval parameter level, get the corresponding configurations
        interval_levels_idxs = {
            parameter: {level: [] for level in interval_levels[parameter]} for parameter in interval_levels.keys()
        }
        for config_idx, config in enumerate(parameter_design):
            for parameter in self.parameter_domain.keys():
                if config[parameter] in interval_levels[parameter]:
                    interval_levels_idxs[parameter][config[parameter]].append(config_idx)
        # get the parameter data, with intervals replaced by samples within those intervals
        parameter_data = deepcopy(parameter_design)
        for parameter in interval_levels_idxs.keys():
            for interval in interval_levels_idxs[parameter].keys():
                num_in_interval = len(interval_levels_idxs[parameter][interval])
                if use_log_uniform is None:
                    interval_linspace = np.linspace(interval[0], interval[1], num=num_in_interval + 1)
                else:
                    if parameter in use_log_uniform:
                        interval_linspace = np.power(10.0, np.linspace(np.log10(interval[0]), np.log10(interval[1]), num=num_in_interval + 1))
                    else:
                        interval_linspace = np.linspace(interval[0], interval[1], num=num_in_interval + 1)

                interval_cuts = [(interval_linspace[i], interval_linspace[i + 1]) for i in
                                 range(len(interval_linspace) - 1)]
                if use_log_uniform is None:
                    values = np.array([np.random.uniform(low=cut[0], high=cut[1]) for cut in interval_cuts])
                else:
                    if parameter in use_log_uniform:
                        values = np.array([log_uniform_10(cut[0], cut[1]) for cut in interval_cuts])
                    else:
                        values = np.array([np.random.uniform(low=cut[0], high=cut[1]) for cut in interval_cuts])

                np.random.shuffle(values)
                for i, config_idx in enumerate(interval_levels_idxs[parameter][interval]):
                    parameter_data[config_idx][parameter] = values[i]

        with open(self.filepath + 'parameter_data.pkl', 'wb') as file:
            pickle.dump(parameter_data, file)
        self.parameter_data = parameter_data
        return parameter_data

    def generate_parameter_configurations(self, covering_array_filename: str, unfilled: str = '*', use_log_uniform: list = None):
        """
        :param covering_array_filename: name of the .csv output from https://srd.sba-research.org/tools/cagen/
        :param unfilled: missing value symbol
        :param use_log_uniform: which parameters to use log-uniform sampling for
        :return:
        """
        parameter_design = self.get_parameter_design(covering_array_filename, unfilled=unfilled)
        parameter_data = self.get_parameter_data(parameter_design, use_log_uniform=use_log_uniform)
        return parameter_data
