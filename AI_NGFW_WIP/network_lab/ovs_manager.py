import os
import subprocess
import time
import requests


API_URL = "http://api:8000/predict"

def run_ovs_cmd(cmd):
    
    full_cmd = f"ovs-ofctl {cmd}"
    print(f"[OVS] running: {full_cmd}")
    os.system(full_cmd)

def setup_baseline_network():
    # nuke the switch state first so we start clean
    print("\n--- wiping switch state ---")
    run_ovs_cmd("del-flows s1")
    
    print("\n--- setting up baseline ---")
    # flood arp or else the hosts cant find each others macs
    run_ovs_cmd("add-flow s1 priority=10,dl_type=0x0806,nw_proto=1,action=flood")
    
    # regular l2 forwarding for normal traffic
    run_ovs_cmd("add-flow s1 priority=1,action=normal")
    
    # apply diffserv qos to prioritize h3
    # mapping to h3 mac and applying dscp 46
    run_ovs_cmd("add-flow s1 priority=20,dl_dst=00:00:00:00:00:03,action=mod_nw_tos:184,normal")
    
    print("\nbaseline done. you can run pingall in mininet to test.")

def block_malicious_host(mac_address):
    # inject a drop flow for the attacker mac. 
    # using priority 50 so it strictly overrides the normal forwarding rule.
    print(f"\n[ENFORCEMENT] zero day detected from {mac_address}")
    run_ovs_cmd(f"add-flow s1 priority=50,dl_src={mac_address},action=drop")
    print(f"host {mac_address} blocked.")

def test_ai_enforcement():
    # simulate sending a zeek log to the api
    print("\n--- querying ai engine ---")
    
    # faking a massive spike in traffic features here to trigger the anomaly detector
    # sending 78 so the backend padder handles it
    simulated_features = [999999.0] * 78 
    
    try:
        response = requests.post(API_URL, json={"features": simulated_features})
        if response.status_code == 200:
            result = response.json()
            print(f"ai response: {result}")
            
            # check if the mse tripped the threshold
            if result.get("threat_detected"):
                # pretend host 2 is the attacker
                block_malicious_host("00:00:00:00:00:02")
        else:
            print(f"api error: {response.status_code}")
    except Exception as e:
        print(f"connection failed. is docker up? err: {e}")

if __name__ == "__main__":
    setup_baseline_network()
    time.sleep(2) # give it a sec to settle
    test_ai_enforcement()
    
    print("\ncurrent flow table:")
    run_ovs_cmd("dump-flows s1")