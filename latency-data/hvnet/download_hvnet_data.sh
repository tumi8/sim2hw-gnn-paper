#!/bin/bash

# load data from HVNet paper dataset
export RSYNC_PASSWORD=m1638129

LATENCY_FILE_NAME='latencies-hvnet.csv'

get_latency_filename() {
  topo_num=$(printf "%02d" "$1")
  echo "${topo_num}-latencies-hvnet.csv"
}

get_latency_filename_preprocessed() {
  topo_num=$(printf "%02d" "$1")
  echo "${topo_num}-latencies-hvnet-preprocessed.csv"
}

load_hvnet_data_repeat() {
  topo_num=$(printf "%02d" "$1")
  repeat=$2

  LATENCY_FILE_NAME=$(get_latency_filename "$topo_num")
  rsync rsync://m1638129@dataserv.ub.tum.de/m1638129/data/${topo_num}/raw_data/latencies-pre-repeat${repeat}.pcap.latency-flows.csv $LATENCY_FILE_NAME
  chmod 644 $LATENCY_FILE_NAME
}

load_hvnet_data() {
  topo=$1

  echo "loading HVNet data for topology $topo"

  LATENCY_FILE_NAME=$(get_latency_filename "$topo")
  REPEAT=1
  load_hvnet_data_repeat $topo $REPEAT
  LATENCY_FILE_SIZE=$(stat -c%s $LATENCY_FILE_NAME)
  while (( LATENCY_FILE_SIZE <  1000 )) && (( REPEAT < 3 )); do
    #echo "repeat $REPEAT did not contain valid data, trying next one..."
    REPEAT=$((REPEAT+1))
    load_hvnet_data_repeat $topo $REPEAT
    LATENCY_FILE_SIZE=$(stat -c%s $LATENCY_FILE_NAME)
  done

  if (( LATENCY_FILE_SIZE <  1000 )); then
    echo "no valid HVNet data for topology $topo, skipping..."
    rm $LATENCY_FILE_NAME
    return 1
  fi
  return 0
}

preprocess_hvnet_data() {
  topo=$1

  LATENCY_FILE_NAME=$(get_latency_filename "$topo")
  LATENCY_FILE_NAME_PREPROCESSED=$(get_latency_filename_preprocessed "$topo")
  echo "preprocessing HVNet data for topology $topo"

  # call python script for preprocessing
  python3 preprocess_hvnet_data.py "$LATENCY_FILE_NAME" "$LATENCY_FILE_NAME_PREPROCESSED"

  # remove original data if preprocessed data exists
  if [[ -e $LATENCY_FILE_NAME_PREPROCESSED ]]; then
    rm $LATENCY_FILE_NAME
  else
    echo "preprocessing failed for topology $topo"
  fi
}

for topo in {0..99}; do
  if [[ ! -e $(get_latency_filename_preprocessed "$topo") ]]; then
    load_hvnet_data $topo
    if [ $? -eq 0 ]; then
      wait  # wait for last preprocessing
      preprocess_hvnet_data $topo &
    fi
  else
    echo "data for topology $topo already exists, skipping..."
  fi
done
