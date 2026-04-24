FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

# install mininet, open vswitch, python tools, and networking utilities.
# iputils-ping provides the ping command for Mininet hosts.
# netcat-openbsd provides nc for TCP connection testing.
# curl and wget are useful for generating HTTP traffic.
# tcpdump is useful for debugging packet captures.
RUN apt-get update && apt-get install -y \
    mininet \
    openvswitch-switch \
    python3 \
    python3-pip \
    iproute2 \
    net-tools \
    iputils-ping \
    netcat-openbsd \
    curl \
    wget \
    tcpdump \
    gnupg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# install zeek from the official repository.
# zeek captures network traffic on the switch interface and writes
# conn.log, which the controller tails for live threat detection.
RUN echo 'deb http://download.opensuse.org/repositories/security:/zeek/xUbuntu_22.04/ /' \
    > /etc/apt/sources.list.d/zeek.list \
    && wget -qO- 'https://download.opensuse.org/repositories/security:/zeek/xUbuntu_22.04/Release.key' \
    | gpg --dearmor > /etc/apt/trusted.gpg.d/zeek.gpg \
    && apt-get update \
    && apt-get install -y zeek \
    && rm -rf /var/lib/apt/lists/*

# add zeek to PATH so the controller and demo scripts can find it
ENV PATH="/opt/zeek/bin:${PATH}"

# install suricata for Phase 6 comparison testing.
# suricata is a signature-based IDS used as a baseline to compare
# against the ML-based anomaly detection approach.
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository -y ppa:oisf/suricata-stable \
    && apt-get update \
    && apt-get install -y suricata \
    && rm -rf /var/lib/apt/lists/*

# update suricata rules so it has signatures to match against
RUN suricata-update || true

RUN pip3 install requests

WORKDIR /workspace
COPY . /workspace

# start the ovs daemon in the background and keep the container alive
CMD service openvswitch-switch start && tail -f /dev/null