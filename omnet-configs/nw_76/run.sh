#!/bin/bash

SIM_TIME_LIMIT=60s

INETPATH=/root/omnet/inet4.5
source /root/omnet/omnetpp-6.0.2/setenv

set -e
set -x

# Run simulation
opp_run -r 0 -m -u Cmdenv -n .:$INETPATH/examples:$INETPATH/src: -l $INETPATH/src/INET \
 --sim-time-limit=$SIM_TIME_LIMIT omnetpp.ini
# Convert vector data to csv
opp_scavetool x -o results/vector-data-omnet.csv results/General-#0.vec
rm results/General-#0.vec

python3 preprocess-latencies.py nw_76
rm results/vector-data-omnet.csv
