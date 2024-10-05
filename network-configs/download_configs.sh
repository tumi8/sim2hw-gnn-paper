#!/bin/bash

set -e

export RSYNC_PASSWORD=m1638129

for i in {0..99}; do
	curr=$(printf "%02d" $i)
	rsync rsync://m1638129@dataserv.ub.tum.de/m1638129/data/$curr/config.json nw-$i.json.tmp
	jq . nw-$i.json.tmp > nw-$i.json
	rm -f nw-$i.json.tmp
	chmod 644 nw-$i.json
done
