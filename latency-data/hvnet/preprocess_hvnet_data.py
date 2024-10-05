import sys
import pandas as pd

LATENCY_FILE_NAME = sys.argv[1]
LATENCY_FILE_NAME_PREPROCESSED = sys.argv[2]

def convert_dstport_to_flow_num(dstport):
    try:
        return int(dstport[3:]) - 1
    except ValueError:
        print(dstport)
        return -1

df = pd.read_csv(LATENCY_FILE_NAME)
df.latency = df.latency / 1e3  # scale to microseconds
df.prets = df.prets / 1e9  # scale to seconds
df.postts = df.postts / 1e9  # scale to seconds
df.dstport = df.dstport.apply(convert_dstport_to_flow_num)  # convert port number to flow number

df.drop(columns=['id', 'srcport'], inplace=True)

df.rename(columns={
    'latency': 'latency_us',
    'prets': 'send_time_s',
    'postts': 'recv_time_s',
    'dstport': 'flow_num'
}, inplace=True)

df.to_csv(LATENCY_FILE_NAME_PREPROCESSED, index=False)
