#!/bin/bash

# -------------------------------------------------------------------
# topology.sh -- create the Mininet topology for the AI-NGFW demo
#
# this script ensures OVS is healthy before starting Mininet.
# inside Docker on WSL2, the container entrypoint starts OVS but
# it can get stuck after failed runs. we forcefully restart it
# every time to guarantee a clean state.
# -------------------------------------------------------------------

echo "--- restarting OVS services ---"

# kill any OVS processes by pid file and by process name.
# using -9 because a stuck ovs-vswitchd will not respond to SIGTERM.
kill -9 $(cat /var/run/openvswitch/*.pid 2>/dev/null) 2>/dev/null
kill -9 $(pgrep ovs) 2>/dev/null
sleep 1

# remove stale socket, pid, and lock files.
# the lock file at .conf.db.~lock~ is critical -- if the old
# ovsdb-server was holding it, the new one cannot start.
rm -f /var/run/openvswitch/db.sock
rm -f /var/run/openvswitch/*.pid
rm -f /var/lib/openvswitch/.conf.db.~lock~

# start ovsdb-server fresh
ovsdb-server --remote=punix:/var/run/openvswitch/db.sock --pidfile --detach

# initialise the database tables
ovs-vsctl --no-wait init

# start ovs-vswitchd fresh
ovs-vswitchd --pidfile --detach

# verify OVS is responding
echo ""
echo "--- checking OVS health ---"
if ovs-vsctl --timeout=5 show > /dev/null 2>&1; then
    echo "OVS is healthy."
    ovs-vsctl --timeout=5 show
else
    echo "ERROR: OVS is not responding. check the container logs."
    exit 1
fi

# wipe any old mininet state
echo ""
echo "--- cleaning up old Mininet state ---"
mn -c 2>/dev/null

# spin up 1 switch and 3 hosts.
# datapath=user is required because the OVS kernel module is not
# available inside Docker on WSL2.
echo ""
echo "--- starting Mininet ---"
mn --topo single,3 --mac --controller=none --switch ovs,datapath=user