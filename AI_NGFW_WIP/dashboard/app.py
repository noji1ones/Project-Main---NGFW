import streamlit as st
import requests
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import onnxruntime as ort
import time

# -- page config --
st.set_page_config(page_title="AI NGFW Monitor", layout="wide")
st.title("AI NGFW Monitor")

# -- constants --
API_URL = "http://api:8000"
THRESHOLD = 0.001936

# the 77 CIC-IDS-2017 feature names in column order.
# these must match the order the scaler was fitted on.
CIC_FEATURE_NAMES = [
    "Destination Port", "Flow Duration", "Total Fwd Packets",
    "Total Backward Packets", "Total Length of Fwd Packets",
    "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean",
    "Fwd Packet Length Std", "Bwd Packet Length Max",
    "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max",
    "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std",
    "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags",
    "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length",
    "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length", "Packet Length Mean",
    "Packet Length Std", "Packet Length Variance", "FIN Flag Count",
    "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
    "ACK Flag Count", "URG Flag Count", "CWE Flag Count",
    "ECE Flag Count", "Down/Up Ratio", "Average Packet Size",
    "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets",
    "Subflow Bwd Bytes", "Init_Win_bytes_forward",
    "Init_Win_bytes_backward", "act_data_pkt_fwd",
    "min_seg_size_forward", "Active Mean", "Active Std", "Active Max",
    "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]


# -------------------------------------------------------------------
# preset payloads for the Traffic Inspector quick-test buttons.
# these mirror the profiles used by run_demo.py and evaluate_performance.py
# so the inspector's results line up with the demo and benchmark output.
# -------------------------------------------------------------------
BENIGN_SHORT = [80, 443, 6, 100]

BENIGN_FULL = [
    80, 50000, 3, 3, 300, 250, 100, 80, 100, 10,
    90, 70, 83, 10, 11000, 120, 16000, 8000, 25000, 5000,
    50000, 16000, 8000, 25000, 5000, 40000, 13000, 7000, 20000, 3000,
    0, 0, 0, 0, 60, 60, 60, 60, 70, 100,
    85, 10, 100, 0, 1, 0, 0, 1, 0, 0,
    0, 1.0, 91, 100, 83, 0, 0, 0, 0, 0,
    0, 3, 300, 3, 250, 29200, 29200, 2, 20,
    0, 0, 0, 0, 0, 0, 0, 0,
]

DDOS_PAYLOAD = [
    80, 12000000, 5000, 3000, 750000, 450000, 500, 100, 150, 120,
    400, 50, 150, 100, 100000, 800, 2400, 1200, 5000, 100,
    12000000, 2400, 1200, 5000, 100, 8000000, 2666, 1500, 6000, 200,
    1, 0, 0, 0, 100000, 60000, 416, 250, 50, 500,
    150, 120, 14400, 3, 5000, 10, 3000, 5000, 0, 0,
    0, 0.6, 150, 150, 150, 0, 0, 0, 0, 0,
    0, 5000, 750000, 3000, 450000, 65535, 65535, 100, 20,
    500, 300, 1000, 100, 3000000, 2000000, 6000000, 1000000,
]

EXFIL_PAYLOAD = [
    4444, 5000000, 200, 10, 500000, 1000, 2500, 2000, 2500, 200,
    100, 100, 100, 0, 100200, 42, 25000, 15000, 50000, 1000,
    5000000, 25000, 15000, 50000, 1000, 200000, 20000, 10000, 40000, 500,
    1, 0, 0, 0, 8000, 400, 40, 2, 100, 2500,
    2400, 900, 810000, 1, 1, 0, 200, 200, 0, 0,
    0, 0.05, 2400, 2500, 100, 0, 0, 0, 0, 0,
    0, 200, 500000, 10, 1000, 65535, 512, 50, 20,
    100, 50, 200, 50, 2000000, 1000000, 4000000, 500000,
]

EXTREME_PAYLOAD = [999999] * 10


# -------------------------------------------------------------------
# model loading (cached so streamlit does not reload on every rerun)
# -------------------------------------------------------------------
@st.cache_resource
def load_model_artifacts():
    """Load all model artifacts once and keep them in memory."""
    scaler = joblib.load("/app/model/scaler.pkl")
    top_features_idx = np.load("/app/model/top_features_idx.npy")
    session = ort.InferenceSession("/app/model/ngfw_ae.onnx")
    background = np.load("/app/model/background_data.npy")
    return scaler, top_features_idx, session, background


# attempt to load artifacts. if this fails the dashboard can still
# show the traffic inspector tab (which calls the API directly).
try:
    scaler, top_features_idx, ort_session, background_data = load_model_artifacts()

    # figure out the human-readable names for the 40 selected features
    selected_feature_names = [CIC_FEATURE_NAMES[i] for i in top_features_idx]

    # the background data might be raw (77 cols) or already processed (40 cols).
    # handle both cases so it works regardless of how it was saved.
    if background_data.shape[1] == 77:
        bg_scaled = scaler.transform(background_data)
        bg_selected = bg_scaled[:, top_features_idx].astype(np.float32)
    elif background_data.shape[1] == len(top_features_idx):
        bg_selected = background_data.astype(np.float32)
    else:
        bg_selected = background_data.astype(np.float32)

    # limit background samples for speed. kernel explainer is O(n) in
    # background size, so 50 samples keeps it responsive for a demo.
    if bg_selected.shape[0] > 50:
        rng = np.random.default_rng(42)
        idx = rng.choice(bg_selected.shape[0], size=50, replace=False)
        bg_selected = bg_selected[idx]

    SHAP_AVAILABLE = True

except Exception as e:
    SHAP_AVAILABLE = False
    st.sidebar.warning(f"Model artifacts not loaded. SHAP tab disabled.\n{e}")


# -------------------------------------------------------------------
# helpers
# -------------------------------------------------------------------
def local_inference(raw_features):
    """Replicate the API pipeline locally so we can get the 40-feature
    vector needed by the SHAP explainer without a second round trip."""
    arr = np.array(raw_features).reshape(1, -1)

    if arr.shape[1] > 77:
        arr = arr[:, :77]
    elif arr.shape[1] < 77:
        padded = np.zeros((1, 77))
        padded[0, :arr.shape[1]] = arr
        arr = padded

    scaled = scaler.transform(arr)
    selected = scaled[:, top_features_idx].astype(np.float32)

    inputs = {ort_session.get_inputs()[0].name: selected}
    reconstruction = ort_session.run(None, inputs)[0]
    mse = float(np.mean(np.square(selected - reconstruction)))

    return mse, selected


def mse_scoring_function(X):
    """Wrapper that SHAP calls. Takes an array of shape (n, 40) and
    returns an array of MSE scores. Each row is a scaled+selected
    feature vector."""
    scores = []
    for row in X:
        row_input = row.reshape(1, -1).astype(np.float32)
        inputs = {ort_session.get_inputs()[0].name: row_input}
        reconstruction = ort_session.run(None, inputs)[0]
        mse = float(np.mean(np.square(row_input - reconstruction)))
        scores.append(mse)
    return np.array(scores)


def parse_feature_input(raw_text):
    """Parse the Traffic Inspector text area. Accepts comma-separated,
    newline-separated, or any mixture of the two. Blank tokens are
    silently dropped."""
    cleaned = raw_text.replace("\n", ",")
    values = []
    for part in cleaned.split(","):
        part = part.strip()
        if part == "":
            continue
        values.append(float(part))
    return values


def payload_to_display_string(payload):
    """Convert a payload list to a neatly formatted string for the text
    area. Uses 10 values per line to keep 77-feature vectors readable."""
    lines = []
    for i in range(0, len(payload), 10):
        chunk = payload[i:i + 10]
        lines.append(", ".join(str(x) for x in chunk))
    return ",\n".join(lines)


def set_preset(payload):
    """on_click callback for the preset buttons. Updates the session
    state so the text area shows the chosen preset on the next rerun,
    and clears any stale result from a previous analysis."""
    st.session_state["inspector_payload"] = payload_to_display_string(payload)
    st.session_state["last_result"] = None


# -------------------------------------------------------------------
# tabs
# -------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Traffic Inspector", "SHAP Explainability", "Live Alerts"])


# ======================== TAB 1: TRAFFIC INSPECTOR ==================
with tab1:
    st.header("Traffic Inspector")
    st.write(
        "Test the AI inference engine by submitting a feature vector. "
        "Pick a preset for a known traffic profile, or paste your own "
        "comma-separated values below."
    )

    # preset buttons. the callback writes to session_state before the
    # text area renders on the next run, so the text area shows the
    # new preset immediately without a rerun dance.
    st.subheader("Quick Test Profiles")
    pc = st.columns(5)
    pc[0].button(
        "Benign (short)",
        on_click=set_preset, args=(BENIGN_SHORT,),
        use_container_width=True,
        help="4-feature minimal payload. Exercises the zero-pad path on the API.",
    )
    pc[1].button(
        "Benign (full)",
        on_click=set_preset, args=(BENIGN_FULL,),
        use_container_width=True,
        help="Realistic 77-feature HTTP browsing flow.",
    )
    pc[2].button(
        "DDoS",
        on_click=set_preset, args=(DDOS_PAYLOAD,),
        use_container_width=True,
        help="High-volume SYN flood profile. Should be flagged as a threat.",
    )
    pc[3].button(
        "Exfiltration",
        on_click=set_preset, args=(EXFIL_PAYLOAD,),
        use_container_width=True,
        help="Asymmetric outbound transfer on an unusual port.",
    )
    pc[4].button(
        "Extreme",
        on_click=set_preset, args=(EXTREME_PAYLOAD,),
        use_container_width=True,
        help="Ten saturation values. Always trips the threshold.",
    )

    # initialise session state on first run
    if "inspector_payload" not in st.session_state:
        st.session_state["inspector_payload"] = "80, 443, 6, 100"
    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None

    st.text_area(
        "Packet Features",
        key="inspector_payload",
        height=140,
        help="Up to 77 numeric values, comma-separated or newline-separated. "
             "Missing values are zero-padded, extras are truncated.",
    )

    ac1, ac2 = st.columns([1, 4])
    analyze_clicked = ac1.button(
        "Analyze", key="analyze_btn", use_container_width=True,
    )

    if analyze_clicked:
        try:
            features_list = parse_feature_input(st.session_state["inspector_payload"])

            if not features_list:
                st.error("No numeric features were parsed. Check the input format.")
            else:
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"features": features_list},
                    timeout=10,
                )
                if response.status_code == 200:
                    result = response.json()
                    st.session_state["last_result"] = {
                        "score": result["malicious_score"],
                        "threat": result["threat_detected"],
                        "features_sent": features_list,
                    }
                else:
                    st.error(f"API returned status {response.status_code}: {response.text}")
                    st.session_state["last_result"] = None

        except requests.exceptions.ConnectionError:
            st.error("Cannot reach the API. Is the API container running?")
        except ValueError as e:
            st.error(f"Feature parse error: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

    # -- result panel --
    if st.session_state["last_result"]:
        result = st.session_state["last_result"]
        score = result["score"]
        threat = result["threat"]
        features_sent = result["features_sent"]

        st.divider()
        st.subheader("Result")

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Anomaly Score (MSE)", f"{score:.6f}")
        mc2.metric("Threshold", f"{THRESHOLD:.6f}")
        mc3.metric("Features Sent", len(features_sent))
        mc4.metric("Classification", "THREAT" if threat else "BENIGN")

        if threat:
            if score >= THRESHOLD:
                ratio = score / THRESHOLD
                st.error(
                    f"THREAT DETECTED -- score {score:.6f} is "
                    f"{ratio:,.1f}x the threshold."
                )
        else:
            ratio_pct = (score / THRESHOLD) * 100 if THRESHOLD > 0 else 0
            st.success(
                f"Traffic appears benign -- score is {ratio_pct:.1f}% of the threshold."
            )

        # visualise the score position relative to the threshold.
        # we clamp the display at 3x the threshold so tiny benign scores
        # are still visible and extreme threat scores don't blow out
        # the bar. scores above 3x simply max the bar out.
        display_max = THRESHOLD * 3
        bar_value = min(score, display_max) / display_max if display_max > 0 else 0
        st.progress(
            bar_value,
            text=f"Score vs 3x threshold scale ({bar_value * 100:.0f}% of window)",
        )

        with st.expander("Show features sent to the model"):
            # show indices alongside CIC-IDS-2017 feature names where
            # we know them, so 'feature 14 = 11000' reads as
            # 'Flow Bytes/s = 11000'.
            names = []
            for i in range(len(features_sent)):
                if i < len(CIC_FEATURE_NAMES):
                    names.append(CIC_FEATURE_NAMES[i])
                else:
                    names.append(f"(feature {i} beyond schema)")

            feat_df = pd.DataFrame({
                "Index": list(range(len(features_sent))),
                "Feature Name": names,
                "Value": features_sent,
            })
            st.dataframe(feat_df, use_container_width=True, hide_index=True)


# ======================== TAB 2: SHAP EXPLAINABILITY =================
with tab2:
    st.header("SHAP Explainability")
    st.write(
        "Pick a scan from the Live Alerts history to see which of the 40 "
        "selected features drove its anomaly score up or down."
    )

    if not SHAP_AVAILABLE:
        st.warning("Model artifacts are not loaded, SHAP analysis is unavailable.")
    else:
        # pull the alert history so the user can select one to explain
        try:
            alerts_response = requests.get(f"{API_URL}/alerts", timeout=5)
            if alerts_response.status_code == 200:
                alerts = alerts_response.json().get("alerts", [])
            else:
                alerts = []
                st.error(f"Failed to fetch alerts: {alerts_response.status_code}")
        except requests.exceptions.ConnectionError:
            alerts = []
            st.error("Cannot reach the API.")

        if not alerts:
            st.info(
                "No scans yet. Send some traffic through the Traffic Inspector "
                "or run the demo to populate the history."
            )
        else:
            # label each scan distinctively so the dropdown is scannable.
            # format: '#N -- timestamp -- OK/THREAT -- score: ...'
            options = []
            for i, a in enumerate(alerts):
                status = "THREAT" if a["threat_detected"] else "OK"
                label = (
                    f"#{len(alerts) - i} -- {a['timestamp']} -- "
                    f"{status} -- score: {a['malicious_score']:.6f}"
                )
                options.append((label, a))

            chosen_label = st.selectbox(
                "Select a scan to explain",
                options=[o[0] for o in options],
                index=0,
            )
            selected_alert = dict(options)[chosen_label]

            sel_score = selected_alert["malicious_score"]
            sel_threat = selected_alert["threat_detected"]

            col1, col2 = st.columns(2)
            col1.metric("Anomaly Score (MSE)", f"{sel_score:.6f}")
            col2.metric("Threat Threshold", f"{THRESHOLD:.6f}")

            if sel_threat:
                st.error(f"This scan was classified as a THREAT (score {sel_score:.4f}).")
            else:
                st.success(f"This scan was classified as benign (score {sel_score:.4f}).")

            if st.button("Explain This Scan", key="explain_btn"):
                try:
                    features_list = selected_alert["features"]

                    # run local inference to get the processed 40-feature vector
                    mse, selected_features = local_inference(features_list)

                    st.write("Running SHAP analysis (this may take a moment)...")

                    explainer = shap.KernelExplainer(mse_scoring_function, bg_selected)
                    shap_values = explainer.shap_values(selected_features, nsamples=100)

                    st.subheader("Feature Importance (Top 15)")

                    sv = shap_values.flatten()
                    feature_importance = pd.DataFrame({
                        "Feature": selected_feature_names,
                        "SHAP Value": sv,
                        "Abs SHAP": np.abs(sv),
                    })
                    feature_importance = feature_importance.sort_values(
                        "Abs SHAP", ascending=False,
                    ).head(15)

                    fig, ax = plt.subplots(figsize=(10, 6))
                    colors = [
                        "#d9534f" if v > 0 else "#5cb85c"
                        for v in feature_importance["SHAP Value"]
                    ]
                    ax.barh(
                        feature_importance["Feature"],
                        feature_importance["SHAP Value"],
                        color=colors,
                    )
                    ax.set_xlabel("SHAP Value (impact on anomaly score)")
                    ax.set_title("Feature Contributions to Anomaly Score")
                    ax.invert_yaxis()
                    plt.tight_layout()
                    st.pyplot(fig)

                    st.caption(
                        "Red bars push the score higher (more anomalous). "
                        "Green bars push it lower (more normal)."
                    )

                    with st.expander("View all 40 feature SHAP values"):
                        full_table = pd.DataFrame({
                            "Feature": selected_feature_names,
                            "Feature Value (scaled)": selected_features.flatten(),
                            "SHAP Value": sv,
                        }).sort_values("SHAP Value", ascending=False, key=abs)
                        st.dataframe(full_table, use_container_width=True, hide_index=True)

                except Exception as e:
                    st.error(f"SHAP analysis failed: {e}")


# ======================== TAB 3: LIVE ALERTS =========================
with tab3:
    st.header("Live Alerts")
    st.write(
        "Recent scan results from the controller pipeline. "
        "Refresh manually, or tick auto-refresh to poll every five seconds."
    )

    cc1, cc2 = st.columns([1, 5])
    auto_refresh = cc1.checkbox("Auto-refresh (5s)", value=False, key="auto_refresh_alerts")
    refresh_clicked = cc2.button("Refresh", key="refresh_btn", use_container_width=False)

    # we fetch either when the user explicitly asks or when auto-refresh
    # is ticked. keeping this condition explicit avoids surprising the
    # user with unrequested API calls while they're navigating.
    should_fetch = auto_refresh or refresh_clicked

    if should_fetch:
        try:
            response = requests.get(f"{API_URL}/alerts", timeout=5)

            if response.status_code == 200:
                alerts = response.json().get("alerts", [])

                if not alerts:
                    st.info(
                        "No alerts yet. The controller has not sent any "
                        "traffic to the API."
                    )
                else:
                    total = len(alerts)
                    threats = sum(1 for a in alerts if a["threat_detected"])
                    benign = total - threats

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Scans", total)
                    col2.metric("Threats", threats)
                    col3.metric("Benign", benign)

                    st.subheader("Anomaly Scores Over Time")
                    df = pd.DataFrame(alerts)
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df.sort_values("timestamp")

                    fig, ax = plt.subplots(figsize=(10, 3))
                    colors = [
                        "#d9534f" if t else "#5cb85c"
                        for t in df["threat_detected"]
                    ]
                    ax.scatter(
                        df["timestamp"], df["malicious_score"],
                        c=colors, s=20, alpha=0.7,
                    )
                    ax.axhline(
                        y=THRESHOLD, color="#f0ad4e", linestyle="--",
                        linewidth=1, label=f"Threshold ({THRESHOLD})",
                    )
                    ax.set_ylabel("Anomaly Score")
                    ax.set_xlabel("Time")
                    ax.legend()
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)

                    st.subheader("Recent Scans")
                    display_df = pd.DataFrame(alerts[:50])
                    display_df["status"] = display_df["threat_detected"].apply(
                        lambda x: "THREAT" if x else "OK"
                    )
                    display_df["malicious_score"] = display_df["malicious_score"].apply(
                        lambda x: f"{x:.6f}"
                    )
                    display_df = display_df[["timestamp", "status", "malicious_score"]]
                    display_df.columns = ["Timestamp", "Status", "Anomaly Score"]
                    st.dataframe(display_df, use_container_width=True, hide_index=True)

            else:
                st.error(f"API returned status {response.status_code}")

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot reach the API. Make sure the API container is running "
                "and the dashboard can connect to http://api:8000."
            )
        except Exception as e:
            st.error(f"Error fetching alerts: {e}")

    # ---------- manual enforcement ----------
    # this section is visible regardless of whether we fetched alerts,
    # so the operator can always queue a block/unblock. the controller
    # picks actions up within ~2 seconds via the /admin/pending_actions
    # endpoint and runs the OVS commands.
    st.divider()
    st.subheader("Manual Enforcement")
    st.write(
        "Force-block an IP, or release a blocked host. Actions are queued "
        "with the API and executed by the controller within a few seconds."
    )

    with st.form(key="manual_block_form", clear_on_submit=True):
        mb1, mb2, mb3 = st.columns([2, 3, 1])
        manual_ip = mb1.text_input(
            "IP to block", placeholder="10.0.0.2", key="manual_block_ip_input",
        )
        manual_reason = mb2.text_input(
            "Reason (optional)",
            placeholder="manual block from dashboard",
            key="manual_block_reason_input",
        )
        submitted = mb3.form_submit_button(
            "Block IP", use_container_width=True,
        )

        if submitted:
            if not manual_ip.strip():
                st.warning("Enter an IP address before clicking Block IP.")
            else:
                try:
                    payload = {
                        "source_ip": manual_ip.strip(),
                        "reason": manual_reason.strip() or "manual block from dashboard",
                    }
                    br = requests.post(
                        f"{API_URL}/admin/block", json=payload, timeout=3,
                    )
                    if br.status_code == 200:
                        st.success(
                            f"Block queued for {manual_ip.strip()}. "
                            f"The controller will install the drop rule shortly."
                        )
                    else:
                        st.error(
                            f"Block request failed: HTTP {br.status_code}: {br.text}"
                        )
                except Exception as e:
                    st.error(f"Block request error: {e}")

    # blocked hosts table with an Unblock button per row
    try:
        block_response = requests.get(f"{API_URL}/blocked", timeout=5)
        if block_response.status_code == 200:
            blocked = block_response.json().get("blocked", [])

            st.subheader("Blocked Hosts")

            if not blocked:
                st.info("No hosts are currently blocked.")
            else:
                # header row
                header = st.columns([2.5, 1.5, 2.5, 1.5, 1.2])
                header[0].markdown("**Blocked At**")
                header[1].markdown("**Source IP**")
                header[2].markdown("**MAC Address**")
                header[3].markdown("**Score**")
                header[4].markdown("**Action**")

                for idx, host in enumerate(blocked):
                    row = st.columns([2.5, 1.5, 2.5, 1.5, 1.2])
                    ts = host.get("timestamp", "")[:19].replace("T", " ")
                    row[0].write(ts)
                    row[1].write(host.get("source_ip", ""))
                    row[2].code(host.get("mac_address", ""), language=None)
                    row[3].write(f"{host.get('anomaly_score', 0):.6f}")

                    if row[4].button(
                        "Unblock",
                        key=f"unblock_btn_{idx}_{host.get('mac_address', '')}",
                        use_container_width=True,
                    ):
                        try:
                            ub = requests.post(
                                f"{API_URL}/admin/unblock",
                                json={"mac_address": host.get("mac_address")},
                                timeout=3,
                            )
                            if ub.status_code == 200:
                                st.success(
                                    f"Unblock queued for {host.get('source_ip')}. "
                                    f"OVS drop rule will be removed shortly."
                                )
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(
                                    f"Unblock failed: HTTP {ub.status_code}: {ub.text}"
                                )
                        except Exception as e:
                            st.error(f"Unblock error: {e}")
    except Exception:
        # do not crash the dashboard if the blocked endpoint is briefly down
        pass

    # ---------- auto-refresh tail ----------
    # keep this at the very end of the tab body so all the UI above has
    # rendered before the five-second sleep triggers the next rerun.
    if auto_refresh:
        time.sleep(5)
        st.rerun()