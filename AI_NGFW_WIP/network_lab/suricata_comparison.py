#!/usr/bin/env python3
"""
suricata_comparison.py -- compare ML-NGFW vs Suricata detection

This script runs during or after the demo to compare what the AI
anomaly detector finds versus what Suricata's signature-based engine
finds. It captures traffic with tcpdump, runs Suricata against the
PCAP, then compares the two systems' results.

Usage (inside network_lab container, after run_demo.py completes):
    python3 suricata_comparison.py

Prerequisites:
    - Suricata must be installed in the container (see updated Dockerfile)
    - The API must be running
    - Traffic must have been generated (run_demo.py or manual)
"""

import os
import sys
import time
import json
import subprocess
import requests
from datetime import datetime

API_URL = "http://api:8000"
PCAP_DIR = "/app/zeek_logs"
PCAP_FILE = os.path.join(PCAP_DIR, "demo_capture.pcap")
SURICATA_LOG_DIR = os.path.join(PCAP_DIR, "suricata_logs")
RESULTS_FILE = os.path.join(PCAP_DIR, "comparison_results.txt")


def check_suricata():
    """Check if Suricata is installed and find its binary."""
    for path in ["/usr/bin/suricata", "/usr/local/bin/suricata"]:
        if os.path.exists(path):
            result = subprocess.run([path, "--build-info"],
                                    capture_output=True, text=True)
            if result.returncode == 0:
                print(f"[SURICATA] found at {path}")
                return path

    print("[ERROR] Suricata is not installed.")
    print("  Add 'suricata' to the network_lab Dockerfile and rebuild.")
    return None


def capture_traffic(interface="s1-eth1", duration=30):
    """Capture live traffic to a PCAP file for Suricata analysis."""
    print(f"\n[CAPTURE] recording {duration}s of traffic on {interface}...")
    os.makedirs(PCAP_DIR, exist_ok=True)

    # remove old capture
    if os.path.exists(PCAP_FILE):
        os.remove(PCAP_FILE)

    proc = subprocess.Popen(
        ["tcpdump", "-i", interface, "-w", PCAP_FILE, "-c", "10000"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )

    time.sleep(duration)
    proc.terminate()
    proc.wait()

    if os.path.exists(PCAP_FILE):
        size = os.path.getsize(PCAP_FILE)
        print(f"[CAPTURE] saved {size:,} bytes to {PCAP_FILE}")
        return True
    else:
        print("[CAPTURE] failed to create PCAP file.")
        return False


def run_suricata_on_pcap(suricata_path, pcap_file):
    """Run Suricata against a PCAP file and parse the results."""
    print(f"\n[SURICATA] analysing PCAP with Suricata...")

    os.makedirs(SURICATA_LOG_DIR, exist_ok=True)

    # clean old logs
    for f in os.listdir(SURICATA_LOG_DIR):
        os.remove(os.path.join(SURICATA_LOG_DIR, f))

    # run suricata in offline mode against the pcap.
    # -r reads a pcap file instead of live capture.
    # -c specifies the config file (required, otherwise suricata fails).
    # -l sets the log output directory.
    # -k none disables checksum validation (needed for virtual interfaces).
    suricata_conf = "/etc/suricata/suricata.yaml"
    if not os.path.exists(suricata_conf):
        print(f"[SURICATA] config file not found at {suricata_conf}")
        return []

    result = subprocess.run(
        [suricata_path, "-r", pcap_file,
         "-c", suricata_conf,
         "-l", SURICATA_LOG_DIR,
         "-k", "none"],
        capture_output=True, text=True, timeout=120,
    )

    if result.returncode != 0:
        print(f"[SURICATA] exited with code {result.returncode}")
        if result.stderr:
            # only print first 500 chars of stderr to avoid flooding
            print(f"[SURICATA] stderr: {result.stderr[:500]}")

    # parse the EVE JSON log if it exists
    eve_log = os.path.join(SURICATA_LOG_DIR, "eve.json")
    fast_log = os.path.join(SURICATA_LOG_DIR, "fast.log")

    alerts = []

    # count how many decoder-level events we filter out. these fire on
    # every packet when running against a pcap captured on a virtual
    # interface: OVS userspace datapath does not compute real TCP
    # checksums, so Suricata flags each packet as having a bad checksum.
    # these are not signature matches on attack patterns and should not
    # be counted as "alerts" in the comparison against the ML system.
    decoder_events_filtered = 0

    if os.path.exists(eve_log):
        with open(eve_log, "r") as f:
            for line in f:
                try:
                    event = json.loads(line.strip())
                    if event.get("event_type") == "alert":
                        sig = event.get("alert", {}).get("signature", "")

                        # skip virtual-interface decoder noise. these all
                        # start with "SURICATA" and name a protocol-level
                        # event (invalid checksum, stream reassembly, etc)
                        # rather than an attack signature.
                        sig_lower = sig.lower()
                        if sig.startswith("SURICATA") and (
                            "checksum" in sig_lower
                            or "decoder" in sig_lower
                            or "stream" in sig_lower
                        ):
                            decoder_events_filtered += 1
                            continue

                        alerts.append({
                            "timestamp": event.get("timestamp", ""),
                            "src_ip": event.get("src_ip", ""),
                            "dest_ip": event.get("dest_ip", ""),
                            "signature": sig,
                            "severity": event.get("alert", {}).get("severity", 0),
                            "category": event.get("alert", {}).get("category", ""),
                        })
                except json.JSONDecodeError:
                    continue

    if os.path.exists(fast_log):
        with open(fast_log, "r") as f:
            fast_lines = f.readlines()
        print(f"[SURICATA] fast.log has {len(fast_lines)} entries")

    if decoder_events_filtered > 0:
        print(f"[SURICATA] filtered out {decoder_events_filtered} decoder-level events")
        print(f"[SURICATA] (virtual-interface artefacts, not attack signatures)")

    print(f"[SURICATA] found {len(alerts)} signature-based attack alerts")

    return alerts, decoder_events_filtered


def get_ml_ngfw_results():
    """Pull the ML-NGFW results from the API."""
    print(f"\n[ML-NGFW] pulling scan results from API...")

    try:
        response = requests.get(f"{API_URL}/alerts", timeout=5)
        if response.status_code == 200:
            alerts = response.json().get("alerts", [])
            print(f"[ML-NGFW] found {len(alerts)} scan results")

            threats = [a for a in alerts if a["threat_detected"]]
            benign = [a for a in alerts if not a["threat_detected"]]
            print(f"[ML-NGFW] threats: {len(threats)}, benign: {len(benign)}")

            return alerts
        else:
            print(f"[ML-NGFW] API error: {response.status_code}")
            return []
    except Exception as e:
        print(f"[ML-NGFW] connection failed: {e}")
        return []


def get_blocked_hosts():
    """Pull the blocked hosts from the API."""
    try:
        response = requests.get(f"{API_URL}/blocked", timeout=5)
        if response.status_code == 200:
            blocked = response.json().get("blocked", [])
            return blocked
        return []
    except Exception:
        return []


def generate_comparison_report(suricata_alerts, ml_alerts, blocked_hosts,
                               decoder_events_filtered=0):
    """Generate a comparison report between the two systems."""
    print("\n" + "=" * 60)
    print("  COMPARISON REPORT: ML-NGFW vs Suricata")
    print(f"  Generated: {datetime.now().isoformat()}")
    print("=" * 60)

    ml_threats = [a for a in ml_alerts if a["threat_detected"]]
    ml_benign = [a for a in ml_alerts if not a["threat_detected"]]

    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("  COMPARISON REPORT: ML-NGFW vs Suricata")
    report_lines.append(f"  Generated: {datetime.now().isoformat()}")
    report_lines.append("=" * 60)

    # ML-NGFW summary
    report_lines.append("\n--- ML-NGFW (Anomaly-Based Detection) ---")
    report_lines.append(f"  Total connections scanned: {len(ml_alerts)}")
    report_lines.append(f"  Threats detected: {len(ml_threats)}")
    report_lines.append(f"  Benign classifications: {len(ml_benign)}")
    report_lines.append(f"  Hosts blocked: {len(blocked_hosts)}")

    if ml_threats:
        scores = [a["malicious_score"] for a in ml_threats]
        report_lines.append(f"  Threat score range: {min(scores):.6f} - {max(scores):.6f}")
        report_lines.append(f"  Mean threat score: {sum(scores)/len(scores):.6f}")

    if blocked_hosts:
        report_lines.append("  Blocked hosts:")
        for bh in blocked_hosts:
            report_lines.append(f"    - {bh['source_ip']} ({bh['mac_address']}) "
                               f"score={bh['anomaly_score']:.6f}")

    print("\n".join(report_lines[-10:]))

    # Suricata summary
    suricata_lines = []
    suricata_lines.append("\n--- Suricata (Signature-Based Detection) ---")
    suricata_lines.append(f"  Signature-based attack alerts: {len(suricata_alerts)}")

    if decoder_events_filtered > 0:
        suricata_lines.append(f"  Decoder-level events filtered: {decoder_events_filtered}")
        suricata_lines.append("  (These are checksum/stream noise from the virtual")
        suricata_lines.append("   OVS interface, not real attack signatures. They")
        suricata_lines.append("   would not fire on a physical network.)")

    if suricata_alerts:
        # group by signature
        sig_counts = {}
        for alert in suricata_alerts:
            sig = alert["signature"]
            sig_counts[sig] = sig_counts.get(sig, 0) + 1

        suricata_lines.append(f"  Unique signatures triggered: {len(sig_counts)}")
        suricata_lines.append("  Alert breakdown:")
        for sig, count in sorted(sig_counts.items(), key=lambda x: -x[1]):
            suricata_lines.append(f"    - {sig}: {count} times")

        # group by source IP
        src_counts = {}
        for alert in suricata_alerts:
            src = alert["src_ip"]
            src_counts[src] = src_counts.get(src, 0) + 1

        suricata_lines.append("  Alerts by source IP:")
        for src, count in sorted(src_counts.items(), key=lambda x: -x[1]):
            suricata_lines.append(f"    - {src}: {count} alerts")
    else:
        suricata_lines.append("  No signature-based attack alerts triggered.")
        suricata_lines.append("  (This is expected for this demo: the synthetic")
        suricata_lines.append("   traffic does not match known Suricata signatures,")
        suricata_lines.append("   whereas the ML system detects volumetric anomalies")
        suricata_lines.append("   regardless of signature coverage.)")

    report_lines.extend(suricata_lines)
    print("\n".join(suricata_lines))

    # comparison analysis
    comparison_lines = []
    comparison_lines.append("\n--- Comparative Analysis ---")

    comparison_lines.append("\nDetection approach differences:")
    comparison_lines.append("  ML-NGFW uses an autoencoder trained on benign traffic to detect")
    comparison_lines.append("  statistical anomalies in flow characteristics. It identifies")
    comparison_lines.append("  traffic that deviates from learned normal patterns regardless")
    comparison_lines.append("  of whether a specific attack signature exists.")
    comparison_lines.append("")
    comparison_lines.append("  Suricata uses predefined rules and signatures to match known")
    comparison_lines.append("  attack patterns in packet payloads and headers. It identifies")
    comparison_lines.append("  specific, catalogued attack types.")

    comparison_lines.append("\nStrengths of ML-NGFW:")
    comparison_lines.append("  - Can detect novel/zero-day attacks with no existing signatures")
    comparison_lines.append("  - Detects volumetric anomalies (DDoS, data exfiltration)")
    comparison_lines.append("  - Provides SHAP explainability for each detection")
    comparison_lines.append("  - Adapts to network-specific normal behaviour via training")

    comparison_lines.append("\nStrengths of Suricata:")
    comparison_lines.append("  - Detects payload-based attacks (SQL injection, XSS)")
    comparison_lines.append("  - Very low false positive rate for known signatures")
    comparison_lines.append("  - Industry-standard rule format (ET, Snort rules)")
    comparison_lines.append("  - No training data required")

    comparison_lines.append("\nComplementary deployment recommendation:")
    comparison_lines.append("  A production NGFW would benefit from running both systems")
    comparison_lines.append("  in parallel. Suricata catches known payload-based attacks")
    comparison_lines.append("  that the autoencoder misses (lower recall on application-")
    comparison_lines.append("  layer attacks), while the autoencoder catches novel")
    comparison_lines.append("  volumetric and behavioural anomalies that have no existing")
    comparison_lines.append("  Suricata signatures.")

    report_lines.extend(comparison_lines)
    print("\n".join(comparison_lines))

    # save the full report
    try:
        with open(RESULTS_FILE, "w") as f:
            f.write("\n".join(report_lines))
        print(f"\nFull report saved to {RESULTS_FILE}")
    except Exception as e:
        print(f"\nCould not save report: {e}")


def run_comparison():
    """Run the full Suricata vs ML-NGFW comparison."""
    print("=" * 60)
    print("  Phase 6: ML-NGFW vs Suricata Comparison")
    print("=" * 60)

    # check if suricata is available
    suricata_path = check_suricata()

    # get ML-NGFW results (these come from the API, populated by the demo)
    ml_alerts = get_ml_ngfw_results()
    blocked = get_blocked_hosts()

    # if suricata is available and we have a pcap, run comparison
    suricata_alerts = []
    decoder_events_filtered = 0
    if suricata_path:
        # check for existing pcap from demo
        if os.path.exists(PCAP_FILE):
            print(f"[PCAP] using existing capture: {PCAP_FILE}")
            suricata_alerts, decoder_events_filtered = run_suricata_on_pcap(
                suricata_path, PCAP_FILE,
            )
        else:
            print("[PCAP] no capture file found.")
            print("  Run this script during or after run_demo.py to have traffic to analyse.")
            print("  Alternatively, capture traffic manually with:")
            print(f"    tcpdump -i s1-eth1 -w {PCAP_FILE} -c 5000 &")
    else:
        print("\n[INFO] Suricata is not installed. Generating comparison report")
        print("  based on ML-NGFW results only, with qualitative Suricata analysis.")

    # generate comparison report regardless
    generate_comparison_report(
        suricata_alerts, ml_alerts, blocked, decoder_events_filtered,
    )


if __name__ == "__main__":
    run_comparison()