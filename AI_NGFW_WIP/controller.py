import os
import time
import json
import requests

API_URL = "http://api:8000/predict"

# path where zeek writes conn.log inside the shared volume
ZEEK_LOG_PATH = "/app/zeek_logs/conn.log"

# how often (seconds) to poll the log file for new lines
POLL_INTERVAL = 1

# mininet with --mac flag assigns sequential macs.
# 10.0.0.1 -> 00:00:00:00:00:01, 10.0.0.2 -> 00:00:00:00:00:02, etc.
# if your topology changes, update this map accordingly.
IP_TO_MAC = {
    "10.0.0.1": "00:00:00:00:00:01",
    "10.0.0.2": "00:00:00:00:00:02",
    "10.0.0.3": "00:00:00:00:00:03",
}


def run_ovs_cmd(cmd):
    """Run an ovs-ofctl command on switch s1."""
    full_cmd = f"ovs-ofctl {cmd}"
    print(f"[OVS] running: {full_cmd}")
    os.system(full_cmd)


def block_host(mac_address, source_ip):
    """Inject a high-priority drop rule for the given MAC address."""
    print(f"\n[ENFORCEMENT] threat detected from {source_ip} ({mac_address})")
    run_ovs_cmd(f"add-flow s1 priority=50,dl_src={mac_address},action=drop")
    print(f"[ENFORCEMENT] host {mac_address} is now blocked on s1.")


def resolve_mac(ip_address):
    """Look up the MAC for a given IP using our static mininet map.
    Falls back to None if the IP is not in the topology."""
    mac = IP_TO_MAC.get(ip_address)
    if mac is None:
        print(f"[WARN] no MAC mapping found for {ip_address}, cannot enforce.")
    return mac


def parse_zeek_header(line):
    """Extract column names from a Zeek #fields header line.
    Returns a list of field names, or None if this is not a header line."""
    if line.startswith("#fields"):
        # format is: #fields\tts\tuid\tid.orig_h\t...
        return line.strip().split("\t")[1:]
    return None


def safe_float(value, default=0.0):
    """Convert a zeek field to float. Zeek uses '-' for unset fields,
    so we catch that and return the default instead of crashing."""
    if value == "-" or value == "(empty)" or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def extract_flag_counts(history_str):
    """Parse Zeek's history field into TCP flag counts.
    Zeek uses single characters: S=SYN, F=FIN, R=RST, P=PSH, A=ACK.
    Uppercase = originator, lowercase = responder. We count both."""
    history = history_str if history_str != "-" else ""
    lower = history.lower()
    return {
        "fin": lower.count("f"),
        "syn": lower.count("s"),
        "rst": lower.count("r"),
        "psh": lower.count("p"),
        "ack": lower.count("a"),
        "urg": 0,  # zeek does not track URG in the history field
    }


def zeek_to_feature_vector(record, fields):
    """Map a single Zeek conn.log record to a 77-element feature vector
    matching the CIC-IDS-2017 column order that the scaler expects.

    We can only populate the fields that Zeek actually provides.
    Everything else stays at 0.0 and the scaler/model handles it.
    This is an inherent limitation of mapping Zeek to CIC-IDS-2017
    and should be discussed in the report."""

    # build a dict for easier field access
    data = {}
    for i, name in enumerate(fields):
        data[name] = record[i] if i < len(record) else "-"

    # pull out the raw values we need
    dst_port = safe_float(data.get("id.resp_p", "-"))
    duration = safe_float(data.get("duration", "-"))
    orig_bytes = safe_float(data.get("orig_bytes", "-"))
    resp_bytes = safe_float(data.get("resp_bytes", "-"))
    orig_pkts = safe_float(data.get("orig_pkts", "-"))
    resp_pkts = safe_float(data.get("resp_pkts", "-"))
    history = data.get("history", "-")

    # derived values (guard against division by zero)
    total_pkts = orig_pkts + resp_pkts
    total_bytes = orig_bytes + resp_bytes

    fwd_pkt_mean = (orig_bytes / orig_pkts) if orig_pkts > 0 else 0.0
    bwd_pkt_mean = (resp_bytes / resp_pkts) if resp_pkts > 0 else 0.0
    overall_pkt_mean = (total_bytes / total_pkts) if total_pkts > 0 else 0.0

    # duration in microseconds (CIC-IDS-2017 uses microseconds, zeek uses seconds)
    duration_us = duration * 1_000_000

    flow_bytes_per_sec = (total_bytes / duration) if duration > 0 else 0.0
    flow_pkts_per_sec = (total_pkts / duration) if duration > 0 else 0.0
    fwd_pkts_per_sec = (orig_pkts / duration) if duration > 0 else 0.0
    bwd_pkts_per_sec = (resp_pkts / duration) if duration > 0 else 0.0

    down_up_ratio = (resp_pkts / orig_pkts) if orig_pkts > 0 else 0.0

    flags = extract_flag_counts(history)

    # build the 77-feature vector in CIC-IDS-2017 column order
    features = [0.0] * 77

    features[0] = dst_port                   # Destination Port
    features[1] = duration_us                 # Flow Duration
    features[2] = orig_pkts                   # Total Fwd Packets
    features[3] = resp_pkts                   # Total Backward Packets
    features[4] = orig_bytes                  # Total Length of Fwd Packets
    features[5] = resp_bytes                  # Total Length of Bwd Packets
    features[6] = fwd_pkt_mean                # Fwd Packet Length Max (estimate)
    features[7] = fwd_pkt_mean                # Fwd Packet Length Min (estimate)
    features[8] = fwd_pkt_mean                # Fwd Packet Length Mean
    # features[9] = 0.0                       # Fwd Packet Length Std (unavailable)
    features[10] = bwd_pkt_mean               # Bwd Packet Length Max (estimate)
    features[11] = bwd_pkt_mean               # Bwd Packet Length Min (estimate)
    features[12] = bwd_pkt_mean               # Bwd Packet Length Mean
    # features[13] = 0.0                      # Bwd Packet Length Std (unavailable)
    features[14] = flow_bytes_per_sec         # Flow Bytes/s
    features[15] = flow_pkts_per_sec          # Flow Packets/s
    # features[16-29] = 0.0                   # IAT stats (unavailable from conn.log)
    features[36] = fwd_pkts_per_sec           # Fwd Packets/s
    features[37] = bwd_pkts_per_sec           # Bwd Packets/s
    features[38] = min(fwd_pkt_mean, bwd_pkt_mean)  # Min Packet Length
    features[39] = max(fwd_pkt_mean, bwd_pkt_mean)  # Max Packet Length
    features[40] = overall_pkt_mean           # Packet Length Mean
    # features[41-42] = 0.0                   # Packet Length Std/Variance (unavailable)
    features[43] = flags["fin"]               # FIN Flag Count
    features[44] = flags["syn"]               # SYN Flag Count
    features[45] = flags["rst"]               # RST Flag Count
    features[46] = flags["psh"]               # PSH Flag Count
    features[47] = flags["ack"]               # ACK Flag Count
    features[48] = flags["urg"]               # URG Flag Count
    # features[49-50] = 0.0                   # CWE/ECE flags (unavailable)
    features[51] = down_up_ratio              # Down/Up Ratio
    features[52] = overall_pkt_mean           # Average Packet Size
    features[53] = fwd_pkt_mean               # Avg Fwd Segment Size
    features[54] = bwd_pkt_mean               # Avg Bwd Segment Size
    # features[55-60] = 0.0                   # Bulk stats (unavailable)
    features[61] = orig_pkts                  # Subflow Fwd Packets
    features[62] = orig_bytes                 # Subflow Fwd Bytes
    features[63] = resp_pkts                  # Subflow Bwd Packets
    features[64] = resp_bytes                 # Subflow Bwd Bytes
    # features[65-76] = 0.0                   # Window/Active/Idle stats (unavailable)

    return features


def query_ai_engine(features):
    """Send the feature vector to the FastAPI inference endpoint.
    Returns the JSON response dict, or None if the request failed."""
    try:
        response = requests.post(API_URL, json={"features": features}, timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"[API] error: status {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"[API] connection failed: {e}")
        return None


def process_connection(record, fields):
    """Process a single Zeek connection record through the full pipeline:
    extract features -> query AI -> enforce if malicious."""

    # figure out who the source is so we can block them later if needed
    orig_h_idx = fields.index("id.orig_h") if "id.orig_h" in fields else None
    source_ip = record[orig_h_idx] if orig_h_idx is not None else "unknown"

    # convert zeek fields to the 77-feature vector
    features = zeek_to_feature_vector(record, fields)

    # send to the AI engine
    result = query_ai_engine(features)
    if result is None:
        return

    score = result.get("malicious_score", 0.0)
    is_threat = result.get("threat_detected", False)

    print(f"[SCAN] src={source_ip} score={score:.6f} threat={is_threat}")

    # if the model flags it, find the MAC and drop the host
    if is_threat:
        mac = resolve_mac(source_ip)
        if mac:
            block_host(mac, source_ip)


def tail_zeek_log():
    """Continuously tail the Zeek conn.log file and process new entries.
    Waits for the file to appear (zeek might not be running yet),
    then reads from the end so we only process live connections."""

    print(f"[CONTROLLER] waiting for zeek log at {ZEEK_LOG_PATH}...")

    # wait for the file to exist before we try reading it
    while not os.path.exists(ZEEK_LOG_PATH):
        time.sleep(POLL_INTERVAL)

    print(f"[CONTROLLER] log file found. starting to tail...")

    fields = None  # will be set once we parse the #fields header

    with open(ZEEK_LOG_PATH, "r") as f:
        # jump to the end of the file so we skip old entries
        f.seek(0, 2)

        while True:
            line = f.readline()

            if not line:
                # no new data yet, wait and try again
                time.sleep(POLL_INTERVAL)
                continue

            line = line.strip()

            # skip blank lines and zeek comment lines (except #fields)
            if not line:
                continue

            if line.startswith("#"):
                # check if this is the #fields header so we know column order
                parsed_fields = parse_zeek_header(line)
                if parsed_fields:
                    fields = parsed_fields
                    print(f"[CONTROLLER] parsed zeek fields: {fields}")
                continue

            # if we have not seen a #fields header yet, we cannot parse data
            if fields is None:
                print("[CONTROLLER] skipping line (no header parsed yet)")
                continue

            # split the tab-separated data line into a record
            record = line.split("\t")

            # sanity check: record should have the same number of columns as fields
            if len(record) != len(fields):
                print(f"[WARN] column mismatch: expected {len(fields)}, got {len(record)}")
                continue

            # run it through the pipeline
            process_connection(record, fields)


if __name__ == "__main__":
    print("=" * 60)
    print("  AI-NGFW Controller")
    print("  Tailing Zeek conn.log -> FastAPI -> OVS Enforcement")
    print("=" * 60)
    tail_zeek_log()