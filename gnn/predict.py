import os
import csv
import sys
from typing import Callable

import yaml
import copy
import time
import json
import torch
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from pprint import pprint
from torch_geometric.data import Data
from collections import defaultdict
import sklearn

import neural_network
from neural_network import import_npz
from parser import Parser
from models import implemented_models
from utils import *


OM2HVNET_PREDICTION_TARGET = 'quantiles'
OM2HVNET_NORMALIZATION = 'log-normalize'

OM2HVNET_FEATURES = get_om2hvnet_features(OM2HVNET_PREDICTION_TARGET)
DENORMALIZE: Callable = get_om2hvnet_denormalization_function(OM2HVNET_NORMALIZATION)

REL_ERR_CSV_FILE = 'results/rel_errs.csv'
TOPOS_USED_FOR_TRAINING = [8, 31, 74, 53, 63, 22, 27, 58, 67, 28, 30, 77, 84, 47, 55, 14, 89, 76, 40, 38, 18, 45, 80, 93, 72, 7, 52, 98, 70, 51, 20, 44, 86, 91, 26, 54, 10, 23, 96, 59, 65, 75, 37, 32, 90, 56, 17, 60, 24, 88, 81, 46, 73, 39, 57, 19, 85, 42, 36, 6, 48, 43, 92, 82, 9, 35, 69, 71, 1, 97, 25]
TOPOS_USED_FOR_TESTING = [94, 21, 66, 68, 49, 87, 4, 2, 34, 50, 13, 11, 5, 83, 12, 0, 15, 61]


def predict_all(args):
    dataset = import_npz(args.dataset)
    assert len(dataset) == len(TOPOS_USED_FOR_TESTING)
    args.num_features = dataset[0].x.shape[1]
    args.num_classes = dataset[0].y.shape[1]
    all_labels = []
    all_preds = []
    device_name = torch.cuda.get_device_name(torch.cuda.current_device()) if args.device == 'cuda' else 'CPU'
    print('Current device before model import:', device_name)
    base_dir = os.path.dirname(args.input_model)
    with open(os.path.join(base_dir, 'config.yml'), 'r') as f:
        model_config = yaml.safe_load(f)
        for k,v in model_config.items():
            # Do not overwrite the model we want to use
            if k == 'input_model':
                continue
            setattr(args, k, v)
    model_import = neural_network.initialize_model(args)

    # Initialize MAPE csv file
    with open(REL_ERR_CSV_FILE, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['topo_num', 'flow_num', 'target', 'label', 'prediction', 'rel_err'])

    prediction_target_names = [f'flow_latency_hvnet_{f}' for f in OM2HVNET_FEATURES]
    rel_err_targets = {}
    predictions_targets = {}
    labels_targets = {}

    rel_err = []
    rel_err_csv_lines = []
    for data_idx, data in enumerate(tqdm(dataset, total=len(dataset))):
        device = torch.device(args.device)
        model = model_import.to(device)
        x, edge_index = data.x, data.edge_index

        pred = neural_network.predict(model, data, device=args.device)
        mask = data.mask

        for mask_vector_idx, mask_vector in enumerate(mask.T):
            target_name = prediction_target_names[mask_vector_idx]
            target_name_short = target_name[len('flow_latency_hvnet_'):]
            curr_flow_num = 0
            for mask_elem_idx, mask_elem in enumerate(mask_vector):
                if mask_elem == 1.0:
                    # Denormalize label and prediction
                    prediction = DENORMALIZE(pred[0][mask_elem_idx][mask_vector_idx].cpu().detach().numpy(), target_name)
                    label = DENORMALIZE(data.y[mask_elem_idx][mask_vector_idx].cpu().detach().numpy(), target_name)

                    if target_name_short in predictions_targets:
                        predictions_targets[target_name_short].append(prediction)
                    else:
                        predictions_targets[target_name_short] = [prediction]
                    if target_name_short in labels_targets:
                        labels_targets[target_name_short].append(label)
                    else:
                        labels_targets[target_name_short] = [label]
                    all_labels.append(label)
                    all_preds.append(prediction)

                    # Calculate errors
                    curr_rel_err = abs(prediction/label - 1)
                    rel_err.append(curr_rel_err)

                    # Append a line to the list for MAPE csv output
                    rel_err_csv_lines.append([
                        TOPOS_USED_FOR_TESTING[data_idx],  # topo number
                        curr_flow_num,
                        target_name_short,
                        label,
                        prediction,
                        curr_rel_err
                    ])

                    if target_name_short in rel_err_targets:
                        rel_err_targets[target_name_short].append(curr_rel_err)
                    else:
                        rel_err_targets[target_name_short] = [curr_rel_err]

                    curr_flow_num += 1

    with open(REL_ERR_CSV_FILE, 'a+') as f:
        writer = csv.writer(f)
        writer.writerows(rel_err_csv_lines)

    # Plot relative errors for each target
    df_rel_errors = pd.DataFrame(rel_err_targets)
    df_rel_errors.boxplot(sym='')
    plt.show()

    # Plot individual labels and predictions
    def plot_label_and_prediction(i):
        df_predictions = pd.DataFrame(predictions_targets).transpose()
        df_labels = pd.DataFrame(labels_targets).transpose()
        plt.plot(df_predictions.index, df_predictions[i], label='predictions')
        plt.plot(df_labels.index, df_labels[i], label='labels')
        plt.legend()
        plt.show()
    # plot_label_and_prediction(16)

    print(f'Mean absolute relative error: {np.mean(rel_err)}, median absolute relative error: {np.median(rel_err)}')
    print(f'MAPE: {mean_absolute_percentage_error(all_labels, all_preds)*100.0}%')
    return None

def main(args):
    res = predict_all(args)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, help="Dataset file (npz)")
    p.add_argument("--output", type=str, default="res.csv", help="File to write results to")
    p.add_argument("--input-model", type=str, default="model.out", help="Model file")
    p.add_argument("--num-input-params", type=int, default=10, help="Number of columns in the input data (x)")
    p.add_argument("--num-classes", type=int, default=1, help="Number of columns in the label data (y)")
    p.add_argument("--submission", action="store_true", help="Set to generate submission")
    p.add_argument("--device", choices=['cpu', 'cuda'], help="Set the device", default='cuda')
    p.add_argument("--hidden-size", type=int, default=64, help="Hidden size")
    p.add_argument("--dropout", type=float, default=0.5, help="Dropout")
    p.add_argument("--nunroll", type=int, default=10, help="Number of unrolls")
    p.add_argument("--model-architecture", choices=[x.__name__ for x in implemented_models], help="Type of torch geometric model acrchitecture to use", default=[x.__name__ for x in implemented_models][0])
    p.add_argument("--no-linear-layer-input", action="store_true", help="Set to NOT use a linear layer before the GRU")
    p.add_argument("--num-layers", type=int, default=1, help="Number of GRU layers")
    p.add_argument("--dropout-gru", type=float, default=.0, help="Dropout used between the GRUs")
    p.add_argument("--mean", type=float, default=.0, help="Mean for z score normalization")
    p.add_argument("--std", type=float, default=.0, help="Standard Deviation for z score normalization")
    args = p.parse_args()
    main(args)

