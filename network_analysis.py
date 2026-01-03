from typing import Literal
import numpy as np
import pandas as pd
import panel as pn
import tensorflow as tf
import keras
from icsa import ICSA, set_seeds
from classification_icsa import load_metadata, MCC
import json
import csv
import pickle
import os
from types import SimpleNamespace
from projector import Projector, OverlayingForward, Feedforward, ConcatenateForward
from plotter import Plotter, MainPlot
from preprocessor import Preprocessor
import warnings
from tqdm import tqdm
import itertools
import pickle
import random


def get_parameter_dictionaries(
        mode: Literal['concatenate', 'overlay'],
        ienhl = [0, 1, 2, 3],
        idnhl = [0, 1, 2, 3],
        cenhl = [0, 1, 2, 3],
        ccat = [0, 1, 2, 3],
        act = ['relu', 'leaky_relu', 'tanh'],
        nhu = [16, 32, 64],
        seed: int = 0):
    """

    :param mode:  A *primary* parameter that determines the defaults of the other parameters, depending on whether the instance/configuration coordinate vectors are concatenated or overlayed in predicting performance
    :param ienhl: list of values for 'instance_encoder_num_hidden_layers'
    :param idnhl: list of values for 'instance_decoder_num_hidden_layers'
    :param cenhl: list of values for 'configuration_encoder_num_hidden_layers'
    :param ccat: list of values for 'performance_num_hidden_layers', if mode = 'concatenate'
    :param act: list of values for 'activation'
    :param nhu: list of values for 'num_hidden_units'
    :return:
    """
    set_seeds(seed=seed)
    if mode == 'concatenate':
        prod = itertools.product(ienhl, idnhl, cenhl, ccat, act, nhu)
    elif mode == 'overlay':
        prod = itertools.product(ienhl, idnhl, cenhl, act, nhu)
    else:
        assert mode in {'concatenate', 'overlay'}
    parameter_dictionaries = []
    for prod_val in prod:
        if mode == 'concatenate':
            ienhl_val, idnhl_val, cenhl_val, ccat_val, act_val, nhu_val = prod_val
            x = dict(
                instance_encoder_num_hidden_layers=ienhl_val,
                instance_decoder_num_hidden_layers=idnhl_val,
                configuration_encoder_num_hidden_layers=cenhl_val,
                performance_num_hidden_layers=ccat_val,
                activation=act_val,
                num_hidden_units=nhu_val,
            )
        elif mode == 'overlay':
            ienhl_val, idnhl_val, cenhl_val, act_val, nhu_val = prod_val
            x = dict(
                instance_encoder_num_hidden_layers=ienhl_val,
                instance_decoder_num_hidden_layers=idnhl_val,
                configuration_encoder_num_hidden_layers=cenhl_val,
                activation=act_val,
                num_hidden_units=nhu_val,
            )
        else:
            assert mode in {'concatenate', 'overlay'}
        parameter_dictionaries.append(x)
    random.shuffle(parameter_dictionaries)
    return parameter_dictionaries


def get_network_results(
        results_path: str,
        feature_data,
        parameter_data,
        performance_data,
        parameter_dictionaries,
        optim_dir: Literal['min', 'max'],
        mode: Literal['concatenate', 'overlay'] = 'concatenate',
        validation_split: float = 0.1,
        epochs: int = 10,
        seed: int = 0):
    """
    Execute network analysis and store the results.

    :param results_path: the path to store the results
    :param feature_data: the feature data
    :param parameter_data: the parameter data
    :param performance_data: the performance data
    :param parameter_dictionaries: a list of parameter dictionaries for the projector model
    :param optim_dir: the optimization direction ('min' or 'max')
    :param mode: the mode of the analysis ('concatenate' or 'overlay') (default: 'concatenate')
    :param validation_split: the ratio of validation data (default: 0.1)
    :param epochs: the number of epochs (default: 10)
    :param seed: the random seed (default: 0)
    :return: None
    """
    results = []  # history object outputs of keras 2
    completed_idxs = []  # completed idxs of parameter_dictionaries
    results_df = []  #  parameter values & performances
    results_exist = os.path.exists(results_path + 'results_df.csv')
    if results_exist:
        with open(results_path + 'results.pkl', 'rb') as file:
            results = pickle.load(file)
        with open(results_path + 'completed_idxs.pkl', 'rb') as file:
            completed_idxs = pickle.load(file)
        results_df = pd.read_csv(results_path + 'results_df.csv')
        results_df = results_df.to_dict(orient='records')
    for x_idx in tqdm(range(len(parameter_dictionaries))):
        x = parameter_dictionaries[x_idx]
        if x_idx in completed_idxs:
            continue
        params = x.copy()
        # num_hidden_units
        params['instance_encoder_num_hidden_units'] = x['num_hidden_units']
        params['instance_decoder_num_hidden_units'] = x['num_hidden_units']
        params['configuration_encoder_num_hidden_units'] = x['num_hidden_units']
        params['performance_num_hidden_units'] = x['num_hidden_units']
        del params['num_hidden_units']
        # activation
        params['instance_encoder_activation'] = x['activation']
        params['instance_decoder_activation'] = x['activation']
        params['configuration_encoder_activation'] = x['activation']
        params['performance_activation'] = x['activation']
        del params['activation']
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            set_seeds(seed=seed)
            analysis = ICSA(optim_dir, mode=mode, **params)  # ICSA with overlapping spaces
            analysis.fit(
                feature_data,
                parameter_data,
                performance_data,
                validation_split=validation_split,
                epochs=epochs,
                verbose=0
            )
            history = analysis._model.history.history
            results.append(history)
            completed_idxs.append(x_idx)
            results_df.extend(
                [{**x, **{'epoch': 1 + epoch}, **{key: results[x_idx][key][epoch] for key in results[x_idx]}}
                 for epoch in range(epochs)]
            )
            with open(results_path + 'results.pkl', 'wb') as file:
                pickle.dump(results, file)
            with open(results_path + 'completed_idxs.pkl', 'wb') as file:
                pickle.dump(completed_idxs, file)
            pd.DataFrame(results_df).to_csv(results_path + 'results_df.csv', index=False)


def load_network_results(results_path: str):
    results_exist = os.path.exists(results_path + 'results_df.csv')
    if results_exist:
        with open(results_path + 'results.pkl', 'rb') as file:
            results = pickle.load(file)
        with open(results_path + 'completed_idxs.pkl', 'rb') as file:
            completed_idxs = pickle.load(file)
        results_df = pd.read_csv(results_path + 'results_df.csv')
        results_df = results_df.to_dict(orient='records')
    return results, completed_idxs, results_df
