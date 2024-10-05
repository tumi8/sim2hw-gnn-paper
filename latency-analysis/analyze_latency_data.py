import os
import sys
import re
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import numpy as np

LATENCY_DATA_DIR = '../latency-data/'
TOPO_CONFIG_DIR = '../network-configs'
RESULTS_DIR = './results'

NUM_TOPOS = 100
MAX_WORKERS = 5

CLEAN_RUN = False

LATENCY_STATS_NAMES = [
    'mean',
    'min',
    'p25',
    'p50',
    'p75',
    'p95',
    'p99',
    'p99.9',
    'p99.99',
    'p99.999'
]

def load_omnet_latencies(topo_num):
    return pd.read_csv(os.path.join(LATENCY_DATA_DIR, 'omnet', f'{topo_num:02d}-latencies-omnet-preprocessed.csv'))


def load_hvnet_latencies(topo_num):
    return pd.read_csv(os.path.join(LATENCY_DATA_DIR, 'hvnet', f'{topo_num:02d}-latencies-hvnet-preprocessed.csv'))


def dict_to_csv(d, filename):
    df = pd.DataFrame.from_dict(d)
    df.to_csv(os.path.join(RESULTS_DIR, filename), index=False)


def stats_to_csv(stats, flow_nums, filename):
    stats_dict = dict(zip(['flow_num'] + LATENCY_STATS_NAMES, [flow_nums] + stats))
    dict_to_csv(stats_dict, filename)


def get_latency_stats_per_flow(topo_num, df_latency):
    flow_nums = []
    mean_lat = []
    min_lat = []
    p25_lat = []
    p50_lat = []
    p75_lat = []
    p95_lat = []
    p99_lat = []
    p99_9_lat = []
    p99_99_lat = []
    p99_999_lat = []

    for f_num in range(df_latency.flow_num.max()+1):
        flow_nums.append(f_num)
        df_lat_f = df_latency[df_latency.flow_num == f_num]

        mean_lat.append(df_lat_f.latency_us.mean())
        min_lat.append(df_lat_f.latency_us.min())
        p25_lat.append(df_lat_f.latency_us.quantile(.25))
        p50_lat.append(df_lat_f.latency_us.quantile(.5))
        p75_lat.append(df_lat_f.latency_us.quantile(.75))
        p95_lat.append(df_lat_f.latency_us.quantile(.95))
        p99_lat.append(df_lat_f.latency_us.quantile(.99))
        p99_9_lat.append(df_lat_f.latency_us.quantile(.999))
        p99_99_lat.append(df_lat_f.latency_us.quantile(.9999))
        p99_999_lat.append(df_lat_f.latency_us.quantile(.99999))

    stats_list = [mean_lat, min_lat, p25_lat, p50_lat, p75_lat, p95_lat, p99_lat, p99_9_lat, p99_99_lat, p99_999_lat]
    assert len(stats_list) == len(LATENCY_STATS_NAMES)

    return stats_list, flow_nums


def analyze_topo_data(topo_num):
    print(f'analyzing topology {topo_num}')

    result_files_exist = os.path.isfile(os.path.join(RESULTS_DIR, f'{topo_num:02d}-latency-stats-hvnet.csv')) \
                         and os.path.isfile(os.path.join(RESULTS_DIR, f'{topo_num:02d}-latency-stats-omnet.csv')) \
                         and os.path.isfile(os.path.join(RESULTS_DIR, f'{topo_num:02d}-latency-corrcoefs.csv'))
    if not CLEAN_RUN and result_files_exist:
        print(f'result files for topology {topo_num} already exist, skipping...')
        return

    # Load and process HVNet data
    try:
        df_latency_hvnet = load_hvnet_latencies(topo_num)
    except FileNotFoundError:
        print(f'no HVNet data for topology {topo_num}, skipping...')
        return
    stats_hvnet, flow_nums_hvnet = get_latency_stats_per_flow(topo_num, df_latency_hvnet)
    stats_to_csv(stats_hvnet, flow_nums_hvnet, f'{topo_num:02d}-latency-stats-hvnet.csv')

    # Load and process OMNeT++ data
    try:
        df_latency_omnet = load_omnet_latencies(topo_num)
    except FileNotFoundError:
        print(f'no OMNeT++ data for topology {topo_num}, skipping...')
        return
    stats_omnet, flow_nums_omnet = get_latency_stats_per_flow(topo_num, df_latency_omnet)
    stats_to_csv(stats_omnet, flow_nums_omnet, f'{topo_num:02d}-latency-stats-omnet.csv')

    stats_corrcoefs = {}
    for i, (stat_hvnet, stat_omnet) in enumerate(zip(stats_hvnet, stats_omnet)):
        corr_val = np.min(np.abs(np.corrcoef(stat_hvnet, stat_omnet)))
        stats_corrcoefs[LATENCY_STATS_NAMES[i]] = [corr_val]
        #print(f'corrcoef {LATENCY_STATS_NAMES[i].upper()}:\t{corr_val:.3f}')
    dict_to_csv(stats_corrcoefs, f'{topo_num:02d}-latency-corrcoefs.csv')

    print(f'finished analysis of topology {topo_num}')


def merge_csvs():
    print('merging csv files...')

    hvnet_files = []
    omnet_files = []
    corrcoefs_files = []

    for filename in os.listdir(RESULTS_DIR):
        file_path = os.path.join(RESULTS_DIR, filename)

        if filename.endswith("-latency-stats-hvnet.csv") or \
           filename.endswith("-latency-stats-omnet.csv") or \
           filename.endswith("-latency-corrcoefs.csv"):
            # Read csv, parse topo_num from filename, and insert topo_num column
            df = pd.read_csv(file_path)
            topo_num = int(filename[:2])
            df.insert(0, 'topo_num', topo_num)

            if filename.endswith("-latency-stats-hvnet.csv"):
                hvnet_files.append(df)
            elif filename.endswith("-latency-stats-omnet.csv"):
                omnet_files.append(df)
            elif filename.endswith("-latency-corrcoefs.csv"):
                corrcoefs_files.append(df)

    hvnet_combined = pd.concat(hvnet_files, ignore_index=True).sort_values(by=['topo_num', 'flow_num'])
    omnet_combined = pd.concat(omnet_files, ignore_index=True).sort_values(by=['topo_num', 'flow_num'])
    corrcoefs_combined = pd.concat(corrcoefs_files, ignore_index=True).sort_values(by='topo_num')

    hvnet_combined.to_csv(os.path.join(RESULTS_DIR, 'latency-stats-hvnet.csv'), index=False)
    omnet_combined.to_csv(os.path.join(RESULTS_DIR, 'latency-stats-omnet.csv'), index=False)
    corrcoefs_combined.to_csv(os.path.join(RESULTS_DIR, 'latency-corrcoefs.csv'), index=False)

    print('csv files merged.')


# Run in parallel for all topos and merge into single csv files
topo_nums = range(NUM_TOPOS)
with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
    executor.map(analyze_topo_data, topo_nums)

merge_csvs()
