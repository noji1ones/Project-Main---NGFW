#!/usr/bin/env python3
"""
run_demo.py -- automated NGFW demonstration

This script creates the Mininet topology, sets up baseline OVS flows,
starts Zeek to capture traffic on the switch interface, and then
generates a mix of benign and malicious traffic so the controller
can detect threats and enforce blocks in real time.

Run this INSTEAD of topology.sh when you want a full end-to-end demo.
The controller.py should be running in a separate terminal first.

Usage (inside the network_lab container):
    python3 run_demo.py
"""

import os
import sys
import time
import subprocess
import requests
from functools import partial

# ---------------------------------------------------------------------------
# mininet imports -- these are available inside the network_lab container
# ---------------------------------------------------------------------------
try:
    from mininet.net import Mininet
    from mininet.topo import SingleSwitchTopo
    from mininet.node import OVSSwitch
    from mininet.cli import CLI
    from mininet.log import setLogLevel
except ImportError:
    print("[ERROR] mininet is not installed in this environment.")
    print("        run this script inside the network_lab container.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

# where zeek should write its logs. the controller is set to tail this path.
ZEEK_LOG_DIR = "/app/zeek_logs"

# how many seconds of benign traffic before the attack starts
BENIGN_WARMUP = 10

# how many seconds of malicious traffic to generate
ATTACK_DURATION = 20

# API base URL reachable from inside the network_lab container.
# we poll /blocked to know when the controller has installed drop rules
# so the final pingAll reflects the enforced state of the switch.
API_BASE_URL = "http://api:8000"


# ---------------------------------------------------------------------------
# OVS management
# ---------------------------------------------------------------------------

def restart_ovs():
    """Forcefully restart OVS services from scratch.

    Inside Docker on WSL2, the container entrypoint starts OVS but
    previous failed Mininet runs can leave it in a stuck state. The
    main culprit is the database lock file at
    /var/lib/openvswitch/.conf.db.~lock~ -- if the old ovsdb-server
    was holding it when it died, the new one cannot start.

    This function kills everything by PID and process name, removes
    all stale files including the lock, and starts fresh."""

    print("[OVS] forcefully restarting OVS services...")

    # kill any OVS processes by pid file and by process name.
    # using -9 because a stuck ovs-vswitchd will not respond to SIGTERM.
    os.system("kill -9 $(cat /var/run/openvswitch/*.pid 2>/dev/null) 2>/dev/null")
    os.system("kill -9 $(pgrep ovs) 2>/dev/null")
    time.sleep(1)

    # remove stale socket, pid, and lock files.
    # the lock file at .conf.db.~lock~ is critical -- if the old
    # ovsdb-server was holding it, the new one cannot start.
    os.system("rm -f /var/run/openvswitch/db.sock")
    os.system("rm -f /var/run/openvswitch/*.pid")
    os.system("rm -f /var/lib/openvswitch/.conf.db.~lock~")

    # start ovsdb-server (the database daemon)
    os.system("ovsdb-server "
              "--remote=punix:/var/run/openvswitch/db.sock "
              "--pidfile --detach")

    # initialise the database tables
    os.system("ovs-vsctl --no-wait init")

    # start ovs-vswitchd (the forwarding daemon)
    os.system("ovs-vswitchd --pidfile --detach")

    # verify OVS is responding
    result = subprocess.run(
        ["ovs-vsctl", "--timeout=5", "show"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print("[OVS] services restarted successfully.")
        print(result.stdout.strip())
    else:
        print("[OVS] ERROR: OVS is not responding after restart.")
        print(f"  stderr: {result.stderr.strip()}")
        print("[OVS] the demo cannot continue without OVS.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# helper functions
# ---------------------------------------------------------------------------

def run_ovs_cmd(cmd):
    """Run an ovs-ofctl command and print it for visibility."""
    full_cmd = f"ovs-ofctl {cmd}"
    print(f"[OVS] {full_cmd}")
    os.system(full_cmd)


def setup_baseline_flows():
    """Install the baseline forwarding rules on switch s1.
    These are the same rules from ovs_manager.py."""
    print("\n--- setting up baseline OVS flows ---")

    # wipe any old state
    run_ovs_cmd("del-flows s1")

    # flood ARP so hosts can resolve each others MAC addresses
    run_ovs_cmd("add-flow s1 priority=10,dl_type=0x0806,nw_proto=1,action=flood")

    # normal L2 forwarding for everything else
    run_ovs_cmd("add-flow s1 priority=1,action=normal")

    # diffserv QoS: prioritize traffic destined for h3 (DSCP 46 = 184 in TOS byte)
    run_ovs_cmd("add-flow s1 priority=20,dl_dst=00:00:00:00:00:03,action=mod_nw_tos:184,normal")

    print("--- baseline flows installed ---\n")


def start_zeek():
    """Start Zeek monitoring switch port s1-eth1.

    With OVS userspace datapath, traffic is visible on the individual
    port interfaces (s1-eth1, s1-eth2, s1-eth3) but NOT on the s1
    bridge interface, 'any', or via multiple -i flags (Zeek only
    allows one). We use s1-eth1 because all demo traffic involves h1:
    benign traffic goes h1 <-> h3, malicious traffic goes h2 -> h1.
    So s1-eth1 sees everything relevant for the demonstration."""

    print("[ZEEK] starting zeek on interface s1-eth1...")

    # make sure the log directory exists and is empty from previous runs
    os.makedirs(ZEEK_LOG_DIR, exist_ok=True)
    os.system(f"rm -f {ZEEK_LOG_DIR}/*.log")

    # bring all interfaces up to be safe
    os.system("ip link set s1-eth1 up")
    os.system("ip link set s1-eth2 up")
    os.system("ip link set s1-eth3 up")

    # check if zeek is available
    zeek_path = None
    for path in ["/opt/zeek/bin/zeek", "/usr/local/zeek/bin/zeek", "/usr/bin/zeek"]:
        if os.path.exists(path):
            zeek_path = path
            break

    if zeek_path is None:
        print("[ZEEK] WARNING: zeek binary not found. skipping zeek startup.")
        print("[ZEEK] the controller will not receive live traffic data.")
        print("[ZEEK] you can still demonstrate OVS enforcement manually.")
        return None

    print(f"[ZEEK] found zeek at {zeek_path}")

    # write zeek's stderr to a file so we can diagnose failures.
    # zeek's stdout is not used, but stderr contains error messages.
    zeek_err_path = os.path.join(ZEEK_LOG_DIR, "zeek_stderr.log")
    zeek_err_file = open(zeek_err_path, "w")

    # start zeek monitoring s1-eth1.
    # -C disables checksum validation (required for virtual interfaces).
    # we do NOT pass 'local' as a script argument -- just let zeek use
    # its default base scripts which include conn.log output.
    zeek_proc = subprocess.Popen(
        [zeek_path, "-i", "s1-eth1", "-C"],
        cwd=ZEEK_LOG_DIR,
        stdout=subprocess.PIPE,
        stderr=zeek_err_file,
    )

    # Popen has duplicated the fd into the child process, so we can
    # close our handle here to avoid leaking a file descriptor.
    zeek_err_file.close()

    # give zeek time to initialize and write its header
    time.sleep(3)

    if zeek_proc.poll() is not None:
        with open(zeek_err_path, "r") as f:
            stderr = f.read()
        print(f"[ZEEK] failed to start: {stderr}")
        return None

    print(f"[ZEEK] running (pid {zeek_proc.pid}), writing logs to {ZEEK_LOG_DIR}")

    # list what files zeek has created so far (for diagnostics)
    files = os.listdir(ZEEK_LOG_DIR)
    print(f"[ZEEK] files in log directory: {files}")

    return zeek_proc


def generate_benign_traffic(net, duration):
    """Generate normal-looking traffic from h1 and h3.
    This includes pings and simple HTTP-like connections."""
    h1 = net.get("h1")
    h3 = net.get("h3")

    print(f"\n[TRAFFIC] generating {duration}s of benign traffic from h1 and h3...")

    # h1 pings h3 in the background (steady, low-rate ICMP)
    h1.cmd(f"ping -c {duration} -i 1 10.0.0.3 &")

    # h3 pings h1 in the background
    h3.cmd(f"ping -c {duration} -i 1 10.0.0.1 &")

    # h1 makes a few short TCP connections to h3 (simulating web browsing)
    # using netcat to open and close connections on common ports
    h3.cmd("while true; do echo 'HTTP 200 OK' | nc -l -p 80 -q 1; done &")
    for i in range(duration // 2):
        h1.cmd("echo 'GET / HTTP/1.1' | nc -w 1 10.0.0.3 80 &")
        time.sleep(2)

    print("[TRAFFIC] benign traffic phase complete.")


def generate_malicious_traffic(net, duration):
    """Generate attack traffic from h2 that produces anomalous flow records.

    The key insight is that the autoencoder was trained on CIC-IDS-2017 data
    where attacks have distinctive flow characteristics: high byte counts,
    unusual packet size ratios, extreme flow rates, and asymmetric transfers.
    Simple nc -z port probes create zero-byte connections that look identical
    to normal short TCP connections in Zeek's conn.log.

    To trigger the anomaly detector, we need connections that actually
    transfer data in patterns the model has not seen during benign training."""
    h2 = net.get("h2")
    h1 = net.get("h1")

    print(f"\n[TRAFFIC] generating {duration}s of MALICIOUS traffic from h2...")

    # set up listeners on h1 so attack connections can complete and transfer data.
    # without listeners, connections fail immediately and zeek logs them as
    # zero-byte rejected connections which look normal to the model.
    print("[TRAFFIC] setting up listeners on h1...")
    h1.cmd("for p in 4444 5555 6666 7777 8888 9999; do "
           "while true; do nc -l -p $p -q 1 > /dev/null; done & done")
    time.sleep(1)

    # attack 1: data exfiltration pattern.
    # send large amounts of random data from h2 to h1 on unusual ports.
    # this creates connections with very high orig_bytes and high flow rates
    # which look very different from normal web browsing.
    print("[TRAFFIC] h2 is exfiltrating data to h1 (high byte volume)...")
    for port in [4444, 5555, 6666]:
        h2.cmd(f"dd if=/dev/urandom bs=50000 count=1 2>/dev/null | nc -w 2 10.0.0.1 {port} &")
        time.sleep(0.5)

    time.sleep(3)

    # attack 2: sustained flood on a single port.
    # send continuous data to one port, creating a long-duration connection
    # with extreme byte counts and packet rates.
    print("[TRAFFIC] h2 is flooding h1:7777 with sustained data...")
    h2.cmd("dd if=/dev/urandom bs=1000 count=500 2>/dev/null | nc -w 5 10.0.0.1 7777 &")
    time.sleep(3)

    # attack 3: many connections with data payloads to unusual ports.
    # unlike nc -z which sends zero bytes, this sends actual payloads
    # which creates connections with non-zero byte counts on unusual ports.
    print("[TRAFFIC] h2 is sending payloads to h1 on multiple ports...")
    for port in [8888, 9999]:
        for i in range(20):
            h2.cmd(f"echo 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA' | nc -w 1 10.0.0.1 {port} &")
            if i % 5 == 0:
                time.sleep(0.3)

    time.sleep(2)

    # attack 4: large ICMP flood (high packet count, long duration).
    # this creates an ICMP flow with extreme packet counts.
    print("[TRAFFIC] h2 is launching a large ping flood against h1...")
    h2.cmd(f"ping -f -c 1000 -s 1400 10.0.0.1 &")

    # wait for remaining attack duration
    remaining = max(0, duration - 12)
    if remaining > 0:
        time.sleep(remaining)

    print("[TRAFFIC] malicious traffic phase complete.")


# ---------------------------------------------------------------------------
# main demo flow
# ---------------------------------------------------------------------------

def run_demo():
    """Run the full end-to-end NGFW demonstration."""
    setLogLevel("info")

    # FIRST: wipe old zeek logs from previous runs BEFORE anything else.
    # this must happen before the controller (running in background) can
    # find an old conn.log and start processing stale data. the controller
    # will wait until zeek creates a fresh conn.log during this demo.
    print("[CLEANUP] removing old zeek logs from previous runs...")
    os.makedirs(ZEEK_LOG_DIR, exist_ok=True)
    os.system(f"rm -rf {ZEEK_LOG_DIR}/*")

    print("=" * 60)
    print("  AI-NGFW Live Demonstration")
    print("  Creating topology -> Starting Zeek -> Generating traffic")
    print("=" * 60)

    # restart OVS from scratch to guarantee a clean state.
    # this prevents hangs caused by leftover state from previous runs.
    restart_ovs()

    # clean up any leftover mininet state
    os.system("mn -c 2>/dev/null")

    # create the topology: 1 switch, 3 hosts, sequential MACs.
    # datapath='user' is required because the OVS kernel module is not
    # available inside Docker on WSL2. this matches topology.sh.
    print("\n[MININET] creating topology: 1 switch, 3 hosts...")
    net = Mininet(
        topo=SingleSwitchTopo(k=3),
        switch=partial(OVSSwitch, datapath='user'),
        controller=None,      # no SDN controller, we manage flows directly
        autoSetMacs=True,      # h1=00:00:00:00:00:01, h2=:02, h3=:03
        autoStaticArp=False,
    )
    net.start()

    # verify connectivity before we begin
    print("\n[MININET] topology started. host details:")
    for h in net.hosts:
        print(f"  {h.name}: IP={h.IP()} MAC={h.MAC()}")

    # install baseline OVS forwarding rules
    setup_baseline_flows()

    # quick connectivity test
    print("\n[MININET] running pingall to verify baseline connectivity...")
    net.pingAll()

    # start zeek to capture traffic
    zeek_proc = start_zeek()

    # start a tcpdump capture for Suricata comparison (Phase 6).
    # this saves a PCAP file that suricata_comparison.py can analyse.
    pcap_file = os.path.join(ZEEK_LOG_DIR, "demo_capture.pcap")
    print(f"[PCAP] starting traffic capture to {pcap_file}...")
    tcpdump_proc = subprocess.Popen(
        ["tcpdump", "-i", "s1-eth1", "-w", pcap_file, "-q"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )

    # give zeek time to fully initialize
    time.sleep(3)

    # send a quick diagnostic ping to verify zeek is capturing
    print("\n[DIAG] sending test ping to verify zeek captures traffic...")
    h1 = net.get("h1")
    h1.cmd("ping -c 3 10.0.0.3")
    time.sleep(5)  # give zeek time to process and write

    # check if conn.log exists now
    conn_log = os.path.join(ZEEK_LOG_DIR, "conn.log")
    if os.path.exists(conn_log):
        size = os.path.getsize(conn_log)
        print(f"[DIAG] conn.log exists ({size} bytes) -- zeek is capturing.")
    else:
        print(f"[DIAG] conn.log does NOT exist yet.")
        # list whatever zeek has written
        files = os.listdir(ZEEK_LOG_DIR)
        print(f"[DIAG] files in {ZEEK_LOG_DIR}: {files}")
        # check if zeek is still running
        if zeek_proc and zeek_proc.poll() is None:
            print(f"[DIAG] zeek process is still running (pid {zeek_proc.pid}).")
        else:
            print(f"[DIAG] zeek process has exited.")
            if os.path.exists(os.path.join(ZEEK_LOG_DIR, "zeek_stderr.log")):
                with open(os.path.join(ZEEK_LOG_DIR, "zeek_stderr.log"), "r") as f:
                    print(f"[DIAG] zeek stderr: {f.read()}")

    print("\n" + "=" * 60)
    print("  PHASE 1: Benign traffic (establishing a normal baseline)")
    print("=" * 60)
    generate_benign_traffic(net, BENIGN_WARMUP)

    # check conn.log status after benign traffic
    time.sleep(2)  # give zeek time to flush
    if os.path.exists(conn_log):
        size = os.path.getsize(conn_log)
        print(f"[DIAG] conn.log after Phase 1: {size} bytes")
    else:
        print(f"[DIAG] conn.log still does not exist after Phase 1.")

    print("\n" + "=" * 60)
    print("  PHASE 2: Attack traffic (h2 is now malicious)")
    print("  Watch the controller terminal for threat detections.")
    print("=" * 60)
    generate_malicious_traffic(net, ATTACK_DURATION)

    # wait for Zeek to flush connection logs and the controller to
    # install its drop rules before we inspect the flow table.
    #
    # Zeek buffers long-duration connections (the sustained flood, the
    # ping flood, the large dd transfers) and only writes them to
    # conn.log when they close or after an internal timeout. without
    # waiting, the most anomalous flows never reach the controller in
    # time to produce any [ENFORCEMENT] action before we dump the
    # flow table and re-test connectivity.
    #
    # instead of a fixed sleep that may fire too early, we poll the
    # API's /blocked endpoint until the controller reports at least
    # one block. this makes the final pingAll an honest verification
    # of enforcement rather than a race condition.
    print("\n[WAIT] waiting for Zeek to flush and controller to enforce...")
    print("[WAIT] watch for [SCAN] and [ENFORCEMENT] lines from the controller.")

    max_wait_seconds = 30
    blocked_count = 0
    enforcement_seen = False

    for _ in range(max_wait_seconds):
        time.sleep(1)
        try:
            resp = requests.get(f"{API_BASE_URL}/blocked", timeout=2)
            if resp.status_code == 200:
                blocked_count = len(resp.json().get("blocked", []))
                if blocked_count > 0 and not enforcement_seen:
                    print(f"[WAIT] controller has blocked {blocked_count} host(s).")
                    enforcement_seen = True
                    # give a small buffer for late-arriving flows so the
                    # flow table reflects all the enforcement that will
                    # happen, not just the first rule that landed.
                    time.sleep(5)
                    break
        except Exception:
            # the API may be briefly unreachable under heavy load.
            # just keep polling until the timeout.
            pass
    else:
        print(f"[WAIT] timed out after {max_wait_seconds}s without seeing any enforcement.")
        print("[WAIT] check the controller terminal for errors or stalled flows.")

    # show the current flow table so you can see the drop rule
    print("\n" + "=" * 60)
    print("  DEMO COMPLETE -- Current OVS flow table:")
    print("=" * 60)
    run_ovs_cmd("dump-flows s1")

    # test connectivity after enforcement
    print("\n[MININET] running pingall to verify h2 is blocked...")
    net.pingAll()

    # drop into the interactive CLI so you can explore further
    print("\n[MININET] dropping into interactive CLI.")
    print("  Type 'pingall' to test connectivity.")
    print("  Type 'h1 ping h3' to test between specific hosts.")
    print("  Type 'exit' to shut down.\n")
    CLI(net)

    # cleanup
    print("\n[CLEANUP] shutting down...")

    if tcpdump_proc and tcpdump_proc.poll() is None:
        tcpdump_proc.terminate()
        tcpdump_proc.wait()
        if os.path.exists(pcap_file):
            size = os.path.getsize(pcap_file)
            print(f"[PCAP] saved {size:,} bytes to {pcap_file}")
        print("[PCAP] stopped.")

    if zeek_proc and zeek_proc.poll() is None:
        zeek_proc.terminate()
        zeek_proc.wait()
        print("[ZEEK] stopped.")

    net.stop()
    print("[MININET] stopped.")


if __name__ == "__main__":
    run_demo()