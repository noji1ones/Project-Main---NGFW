FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

# install mininet, open vswitch, and python tools
RUN apt-get update && apt-get install -y \
    mininet \
    openvswitch-switch \
    python3 \
    python3-pip \
    iproute2 \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install requests

WORKDIR /workspace
COPY . /workspace

# start the ovs daemon in the background and keep the container alive
CMD service openvswitch-switch start && tail -f /dev/null