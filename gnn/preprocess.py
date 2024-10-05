import json
import random
import re
import os
from typing import Callable

import networkx as nx
import pandas as pd
import numpy as np
from scipy.stats import gamma
from multiprocessing import Pool
from tqdm import tqdm
from parser import Parser
from utils import *


OM2HVNET_PREDICTION_TARGET = 'quantiles'
OM2HVNET_NORMALIZATION = 'log-normalize'

NUM_TOPOS = 100
MAX_SAMPLES = 10000
SEED = 1234
NUM_PROCESSES = 6

NO_DATA_TOPOS = [33, 78, 79]
OUTLIER_TOPOS = [3, 16, 29, 41, 62, 64, 95, 99]
SKIPPED_TOPOS = NO_DATA_TOPOS + OUTLIER_TOPOS

TOPO_CONFIG_DIR = '../network-configs'
GAMMA_PARAMS_FILE = '../hvnet-flow-gamma-params/gamma-params.json'
LATENCY_STATS_FILE_HVNET = '../latency-analysis/results/latency-stats-hvnet.csv'
LATENCY_STATS_FILE_OMNET = '../latency-analysis/results/latency-stats-omnet.csv'

DATASET_DIR = f'data/input-datasets/predict-{OM2HVNET_PREDICTION_TARGET}-{OM2HVNET_NORMALIZATION}'
TRAIN_DATASET_FILE = f'{DATASET_DIR}/training/train.npz'
TEST_DATASET_FILE = f'{DATASET_DIR}/test.npz'

os.makedirs(f'{DATASET_DIR}/training/', exist_ok=True)

OM2HVNET_FEATURES = get_om2hvnet_features(OM2HVNET_PREDICTION_TARGET)
NORMALIZE: Callable = get_om2hvnet_normalization_function(OM2HVNET_NORMALIZATION)


def get_feature(df_stats, topo_num, flow_num, feature):
    try:
        feature = df_stats[(df_stats.topo_num == topo_num) & (df_stats.flow_num == flow_num)][feature].iloc[0]
    except (IndexError, KeyError):
        raise ValueError(f'No data found for {feature} of flow {flow_num} in topology {topo_num}')
    return feature


class OM2HVNetParser(Parser):

    def process_data(self, data):

        topo_num, gamma_params, topo_config, df_stats_hvnet, df_stats_omnet = data

        G = nx.Graph()

        parameter_template = {k: float('nan') for k in self.node_parameters}
        # Initialize all values with NaN, this allows to skip unset values during normalization

        def get_params(kwargs):
            params = dict(parameter_template)
            params.update((k, kwargs[k]) for k in set(kwargs).intersection(params))
            return params

        for server in topo_config['server']:
            G.add_node(f"{server['name']}-{server['name']}", **get_params({
                'ntype': self.node_types.SelfLink
            }))

        for link in topo_config['link']:
            G.add_node(f"{link['start']}-{link['stop']}", **get_params({
                'ntype': self.node_types.Link,
                'hvnet_link_rate': link['rate'],  # Mbit/s
                'omnet_link_rate': 1e3  # hardcode to 1Gbps
            }))

        for flow in topo_config['flow']:
            rate = flow['rate']
            rate_b_s = rate * 125
            flow_num = int(flow['name'][1:])
            flow_params = {
                'ntype': self.node_types.Flow,
                'flow_rate': rate / 1e3,  # convert to Mbit/s (same as link rate)
                'flow_gamma_shape': gamma_params[str(float(rate_b_s))]['shape'],
                'flow_gamma_scale': gamma_params[str(float(rate_b_s))]['scale']
            }

            # Add OMNeT++ features
            for feature in OM2HVNET_FEATURES:
                flow_params[f'flow_latency_omnet_{feature}'] = get_feature(df_stats_omnet, topo_num, flow_num, feature)

            # Add HVNet features
            for feature in OM2HVNET_FEATURES:
                flow_params[f'flow_latency_hvnet_{feature}'] = get_feature(df_stats_hvnet, topo_num, flow_num, feature)

            G.add_node(flow['name'], **get_params(flow_params))

        for flow in topo_config['flow']:
            for i, p in enumerate(flow['hops']):
                path_node = f"p{flow['name'][1:]}-{i}"  # compute index of path node
                G.add_node(path_node, **get_params({'ntype': self.node_types.Path, 'path': i}))
                G.add_edge(path_node, flow['name'])
                if i == len(flow['hops']) - 1:
                    G.add_edge(path_node, f"{p}-{p}")
                else:
                    G.add_edge(path_node, f"{p}-{flow['hops'][i+1]}")
        return G


if __name__ == "__main__":
    # Node attributes for all node types
    # encoding: values = 0 are encoded as scalar, values > 0 are one-hot encoded to the length of value
    # is_y: set to true if we want to predict this value
    # mask: only required if is_y, sets a list of node types as mask fpr the loss function
    node_parameters = {
        'ntype': {
            'encoding': len(OM2HVNetNodeType) + 1, 'is_y': False
        }, 'flow_rate': {
            'encoding': 0, 'is_y': False,
            'normalization': lambda x: NORMALIZE(x, 'flow_rate')
        }, 'flow_gamma_shape': {
            'encoding': 0, 'is_y': False,
            'normalization': lambda x: NORMALIZE(x, 'flow_gamma_shape')
        }, 'flow_gamma_scale': {
            'encoding': 0, 'is_y': False,
            'normalization': lambda x: NORMALIZE(x, 'flow_gamma_scale')
        }, 'hvnet_link_rate': {
            'encoding': 0, 'is_y': False,
            'normalization': lambda x: NORMALIZE(x, 'hvnet_link_rate')
        }, 'omnet_link_rate': {
            'encoding': 0, 'is_y': False,
            'normalization': lambda x: NORMALIZE(x, 'omnet_link_rate')
        }
    }

    # Add OMNeT++ features
    for f in OM2HVNET_FEATURES:
        param_name = f'flow_latency_omnet_{f}'
        node_parameters[param_name] = {
            'encoding': 0, 'is_y': False,
            'normalization': lambda x: NORMALIZE(x, param_name)
        }

    # Add HVNet features
    for f in OM2HVNET_FEATURES:
        param_name = f'flow_latency_hvnet_{f}'
        node_parameters[param_name] = {
            'encoding': 0, 'is_y': True,
            'normalization': lambda x: NORMALIZE(x, param_name),
            'mask': [OM2HVNetNodeType.Flow]
        }

    with open(GAMMA_PARAMS_FILE) as f:
        gamma_params = json.load(f)

    df_stats_hvnet = pd.read_csv(LATENCY_STATS_FILE_HVNET)
    df_stats_omnet = pd.read_csv(LATENCY_STATS_FILE_OMNET)

    def calc_matrix_for_topo(i):
        if i in SKIPPED_TOPOS:
            return i, None

        with open(f'{TOPO_CONFIG_DIR}/nw-{i}.json') as f:
            topo_config = json.load(f)

        parser = OM2HVNetParser(
            node_types=OM2HVNetNodeType,
            node_parameters=node_parameters
        )
        processed = parser.import_raw((i, gamma_params, topo_config, df_stats_hvnet, df_stats_omnet))
        matrix = parser.graph2matrix(processed, lambda x: x)
        return i, matrix

    with Pool(processes=NUM_PROCESSES) as p:
        topo_matrices = list(tqdm(p.imap(calc_matrix_for_topo, [i for i in range(NUM_TOPOS)]), total=NUM_TOPOS))
        for i, tm in topo_matrices:
            if tm is None:
                print(f'topo {i} was skipped (no data or outlier topo).')
        topo_matrices = [(i, tm) for i, tm in topo_matrices if tm is not None]

    parser = OM2HVNetParser(
        node_types=OM2HVNetNodeType,
        node_parameters=node_parameters
    )

    # Split data into train and test datasets
    random.Random(SEED).shuffle(topo_matrices)
    split_index = int(len(topo_matrices) * 0.8)  # split at 80 %
    topo_matrices_train = [elem for _, elem in topo_matrices[:split_index]]
    topo_indices_train = [i for i, _ in topo_matrices[:split_index]]
    topo_matrices_test = [elem for _, elem in topo_matrices[split_index:]]
    topo_indices_test = [i for i, _ in topo_matrices[split_index:]]
    print(f'topos used for training: {topo_indices_train}')
    print(f'topos used for testing: {topo_indices_test}')
    parser.export_data(np.array(topo_matrices_train, dtype=object), TRAIN_DATASET_FILE)
    parser.export_data(np.array(topo_matrices_test, dtype=object), TEST_DATASET_FILE)
