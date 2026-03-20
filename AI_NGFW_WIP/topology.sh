#!/bin/bash

# wipe any old mininet junk in case it hung on the last exit
mn -c 

# spin up 1 switch and 3 hosts. using simple macs so its easier to read later.
# killing the default controller so our python script can handle the routing rules.
mn --topo single,3 --mac --controller=none --switch ovs,datapath=user