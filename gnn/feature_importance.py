import os.path as osp
import pandas as pd

import os
import yaml
import csv
import argparse
import signal
import numpy as np
import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch_geometric.transforms as T
import neural_network
from neural_network import import_npz, import_multiple_npz
import random
import copy
import tqdm
from typing import Callable
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from utils import *


OM2HVNET_PREDICTION_TARGET = 'quantiles'
OM2HVNET_NORMALIZATION = 'log-normalize'

OM2HVNET_FEATURES = get_om2hvnet_features(OM2HVNET_PREDICTION_TARGET)
DENORMALIZE: Callable = get_om2hvnet_denormalization_function(OM2HVNET_NORMALIZATION)


def permute_idx(dataset, index_to_permute, index_stop=None):
    dataset_copy = copy.deepcopy(dataset)
    for idx, data in enumerate(dataset_copy):
        print(idx, data)
        # see https://github.com/numpy/numpy/issues/18206
        d = data.x.cpu().numpy()
        if index_stop:
            np.random.shuffle(d[:, index_to_permute:index_stop])
        else:
            np.random.shuffle(d[:,index_to_permute])
        data.x = torch.from_numpy(d)
    return dataset_copy


def permute_idx_per_node_type(dataset, parameter, index_to_permute, index_stop=None):
    ntype_idx = parameter['ntype']

    dataset_copy = copy.deepcopy(dataset)
    for idx, data in enumerate(dataset_copy):
        print(idx, data)

        node_types = torch.unique(data.x[:, ntype_idx[0]:ntype_idx[1]], dim=0)

        for nt in node_types:
            nt_mask = (data.x[:, ntype_idx[0]:ntype_idx[1]] == nt).all(dim=1)
            elems_to_shuffle = data.x[nt_mask]

            if index_stop:
                elems_to_shuffle = elems_to_shuffle[:, index_to_permute:index_stop]
                shuffled_elems = elems_to_shuffle[torch.randperm(elems_to_shuffle.size(0))]
                data.x[nt_mask, index_to_permute:index_stop] = shuffled_elems
            else:
                elems_to_shuffle = elems_to_shuffle[:, index_to_permute]
                shuffled_elems = elems_to_shuffle[torch.randperm(elems_to_shuffle.size(0))]
                data.x[nt_mask, index_to_permute] = shuffled_elems

    return dataset_copy


def import_node_parameters(file, data_sample):

    with open(file, 'r') as f:
        node_parameters = yaml.safe_load(f)

    index = 0
    out = {}

    for name, values in node_parameters.items():
        if 'is_y' in values and values['is_y']:
            continue
        if values['encoding'] == 0:
            print(f'{name}: {index}')
            if not values.get('ignore', False):
                out[name] = [index]
            index += 1
        else:
            print(f'{name}: {index} - {index + values["encoding"]}')
            if not values.get('ignore', False):
                out[name] = [index, index + values['encoding']]
            index += values['encoding']

    _, num_features = data_sample.shape
    print(f'comparing {num_features} and {index}')
    assert num_features == index

    return out


def run_feature_importance(args):

    dataset = import_npz(args.dataset)
    base_dir = os.path.dirname(args.input_model)

    with open(os.path.join(base_dir, 'config.yml'), 'r') as f:
        model_config = yaml.safe_load(f)
        for k, v in model_config.items():

            if k in vars(args):  # Do not overwrite args
                continue
            setattr(args, k, v)

    if 'num_features' not in model_config.keys():
        setattr(args, 'num_features', dataset[0].x.shape[1])
    if 'num_classes' not in model_config.keys():
        setattr(args, 'num_classes', dataset[0].y.shape[1])

    model_import = neural_network.initialize_model(args)
    print('Dataset loaded')

    parameter = import_node_parameters(args.columns, data_sample=dataset[0].x)

    res_all = []
    random.seed()

    baseline = None

    for epoch in range(args.num_epochs):
        res = []

        all_datasets = {}

        for name, indices in tqdm.tqdm(parameter.items(), total=len(parameter), ncols=0, leave=False):
            if name == 'ntype':
                # permute amongst all node types for the 'ntype' parameter (otherwise, no shuffling will be done)
                all_datasets[name] = permute_idx(dataset, *indices)
            else:
                all_datasets[name] = permute_idx_per_node_type(dataset, parameter, *indices)

        if baseline is None:
            all_datasets['baseline'] = dataset

        for one_dataset_indx, (name, one_dataset) in enumerate(all_datasets.items()):
            #res_one_dataset = []
            #print(one_dataset.__dict__)
            #for data_indx, data in enumerate(tqdm.tqdm(one_dataset, total=1000, desc=f'{name}', ncols=0, leave=False)):
            model = model_import.to(args.device)
            # fails bc multiple time steps, not directly x
            #print(f'ONE DATASET: {one_dataset}')
            #x, edge_index, mask = data.x, data.edge_index, data.mask

            pred = neural_network.predict(model, one_dataset[0], device=args.device)

            # NOTE: this only works when all masks are the same
            data = one_dataset[0]
            mask = data.mask.T[0].to(torch.bool)
            mlabels = data.y[mask]
            moutput = pred[0][mask]

            # NOTE: this only works when the denormalization function is the same for all prediction targets
            mlabels_denorm = DENORMALIZE(mlabels.detach().cpu())
            moutput_denorm = DENORMALIZE(moutput.detach().cpu())

            mape_data = mean_absolute_percentage_error(mlabels_denorm, moutput_denorm)

            res.append(mape_data)
        
        if baseline is None:
            baseline = res[-1]
        else:
            res.append(baseline)
        res_all.append(res)
        print("Res: {}".format(res))

        df_all = pd.DataFrame(res_all, columns=list(parameter.keys()) + ['baseline'])
        print(df_all)
        df_all.to_csv(args.output, index=False)


def main(args):

    if args.device == 'cuda':
        try:
            torch.cuda.current_device()
        except Exception as e:
            print(e)
            print('Fallback to cpu')
            args.device = 'cpu'

    res = run_feature_importance(args)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True, help="Dataset file (npz)")
    p.add_argument("--output", type=str, default="res.csv", help="File to write results to")
    p.add_argument("--input-model", type=str, required=True, help="Model file")
    p.add_argument("--num-epochs", type=int, default=1, help="Number of epochs to repeat randomized permutation")
    p.add_argument("--device", choices=['cpu', 'cuda'], help="Set the device", default='cuda')
    p.add_argument("--columns", required=True, help='Yaml file with node parameters.')
    args = p.parse_args()
    main(args)
