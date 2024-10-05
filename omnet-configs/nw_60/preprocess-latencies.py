import pandas as pd
import os
import sys

TOPO_NUM = int(sys.argv[1].split('_')[-1])

FLOWMAP_FILE = './flowmap.csv'
OMNET_RESULTS_FOLDER = './results'
OMNET_VECTOR_DATA_FILE_NAME = 'vector-data-omnet.csv'
OMNET_PREPROCESSED_CSV_FILE = f'{TOPO_NUM:02d}-latencies-omnet-preprocessed.csv'

OMNET_LATENCY_DATA_DIR = '../../latency-data/omnet/'


def load_omnet_latencies():
    _flowmap = pd.read_csv(FLOWMAP_FILE)
    df_omnet = pd.read_csv(os.path.join(OMNET_RESULTS_FOLDER, OMNET_VECTOR_DATA_FILE_NAME))

    df_omnet['_module'] = df_omnet['module'].str.rpartition('.')[0]  # fix module name for merge
    df_omnet = df_omnet.merge(_flowmap[['recv_module', 'flow']], left_on='_module', right_on='recv_module')

    df_latency_list = []
    for f in _flowmap.flow.unique():
        f_num = int(f[1:])
        latency_vectors = df_omnet.loc[
            (df_omnet['flow'] == f)
            & (df_omnet['name'] == 'meanBitLifeTimePerPacket:vector')
            & (df_omnet['type'] == 'vector')
            ]
        vectimes = pd.Series(latency_vectors['vectime'].values[0].split(), name='recv_time_s').astype(float)
        vecvalues = pd.Series(latency_vectors['vecvalue'].values[0].split(), name='latency_us').astype(float)

        df_latency_f = pd.concat([vecvalues, vectimes], axis=1).assign(flow_num=f_num)
        df_latency_list.append(df_latency_f)

    df = pd.concat(df_latency_list)

    df.latency_us = df.latency_us * 1e6  # scale to microseconds

    # store preprocessed dataframe to csv
    os.makedirs(OMNET_LATENCY_DATA_DIR, exist_ok=True)
    df.to_csv(os.path.join(OMNET_LATENCY_DATA_DIR, OMNET_PREPROCESSED_CSV_FILE), index=False)


load_omnet_latencies()
