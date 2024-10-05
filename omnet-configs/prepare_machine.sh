#!/bin/bash

set -e
set -x

# Install OMNeT++ and INET
DEBIAN_FRONTEND=noninteractive apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential clang lld gdb bison flex perl python3 python3-pip python3-venv \
    libxml2-dev zlib1g-dev doxygen graphviz libwebkit2gtk-4.0-37 xdg-utils libgraphviz-dev tig
DEBIAN_FRONTEND=noninteractive apt-get install -y python3-numpy python3-pandas python3-matplotlib python3-seaborn

python3 -m pip install --break-system-packages posix_ipc
mkdir -p /usr/share/desktop-directories

mkdir /root/omnet
pushd /root/omnet

# Install OMNeT++ 6.0.2
wget https://github.com/omnetpp/omnetpp/releases/download/omnetpp-6.0.2/omnetpp-6.0.2-linux-x86_64.tgz
tar -xzf omnetpp-6.0.2-linux-x86_64.tgz
rm omnetpp-6.0.2-linux-x86_64.tgz
pushd omnetpp-6.0.2
source setenv -f
./configure WITH_TKENV=no WITH_QTENV=no WITH_OSG=no WITH_OSGEARTH=no
make -j
popd

# Download INET 4.5
wget https://github.com/inet-framework/inet/releases/download/v4.5.2/inet-4.5.2-src.tgz
tar -xzf inet-4.5.2-src.tgz
rm inet-4.5.2-src.tgz
pushd inet4.5

# Apply patches to INET
pushd src/inet/transportlayer
sed -i 's/EPHEMERAL_PORTRANGE_START = [0-9]\+/EPHEMERAL_PORTRANGE_START = 5000/' udp/Udp.h
sed -i 's/EPHEMERAL_PORTRANGE_END = [0-9]\+/EPHEMERAL_PORTRANGE_END = 10000/' udp/Udp.h
sed -i 's/EPHEMERAL_PORTRANGE_START = [0-9]\+/EPHEMERAL_PORTRANGE_START = 5000/' tcp/Tcp.h
sed -i 's/EPHEMERAL_PORTRANGE_END = [0-9]\+/EPHEMERAL_PORTRANGE_END = 10000/' tcp/Tcp.h
popd

# Build INET
source setenv -f
make makefiles
make -j
popd

popd
