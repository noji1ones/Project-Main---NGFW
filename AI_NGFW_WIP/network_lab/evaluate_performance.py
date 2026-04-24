#!/usr/bin/env python3
"""
evaluate_performance.py -- Phase 6 evaluation benchmarks

Measures inference latency, throughput under load, and detection
accuracy across different traffic profiles. Run this inside the
network_lab container while the API is running.

Usage (inside network_lab container):
    python3 evaluate_performance.py

Results are printed to stdout and saved to /app/zeek_logs/evaluation_results.txt
"""

import time
import json
import statistics
import requests
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

API_URL = "http://api:8000/predict"

# ---------------------------------------------------------------------------
# test payloads -- covering benign, DDoS, exfiltration, and edge cases
# ---------------------------------------------------------------------------

# benign: simple web browsing (port 80, small values)
BENIGN_PAYLOAD = {"features": [80, 443, 6, 100]}

# benign: full 77-feature realistic normal HTTP connection.
# values represent a single short web request: port 80, ~50ms duration,
# a few packets each way, small byte counts, normal TCP flags.
# all values are deliberately small to stay within the benign distribution
# the autoencoder learned during training.
BENIGN_FULL = {"features": [
    80, 50000, 3, 3, 300, 250, 100, 80, 100, 10,
    90, 70, 83, 10, 11000, 120, 16000, 8000, 25000, 5000,
    50000, 16000, 8000, 25000, 5000, 40000, 13000, 7000, 20000, 3000,
    0, 0, 0, 0, 60, 60, 60, 60, 70, 100,
    85, 10, 100, 0, 1, 0, 0, 1, 0, 0,
    0, 1.0, 91, 100, 83, 0, 0, 0, 0, 0,
    0, 3, 300, 3, 250, 29200, 29200, 2, 20,
    0, 0, 0, 0, 0, 0, 0, 0
]}

# DDoS attack: high packet counts, high byte volumes, many SYN flags
DDOS_PAYLOAD = {"features": [
    80, 12000000, 5000, 3000, 750000, 450000, 500, 100, 150, 120,
    400, 50, 150, 100, 100000, 800, 2400, 1200, 5000, 100,
    12000000, 2400, 1200, 5000, 100, 8000000, 2666, 1500, 6000, 200,
    1, 0, 0, 0, 100000, 60000, 416, 250, 50, 500,
    150, 120, 14400, 3, 5000, 10, 3000, 5000, 0, 0,
    0, 0.6, 150, 150, 150, 0, 0, 0, 0, 0,
    0, 5000, 750000, 3000, 450000, 65535, 65535, 100, 20,
    500, 300, 1000, 100, 3000000, 2000000, 6000000, 1000000
]}

# data exfiltration: high forward bytes, unusual port, asymmetric transfer
EXFIL_PAYLOAD = {"features": [
    4444, 5000000, 200, 10, 500000, 1000, 2500, 2000, 2500, 200,
    100, 100, 100, 0, 100200, 42, 25000, 15000, 50000, 1000,
    5000000, 25000, 15000, 50000, 1000, 200000, 20000, 10000, 40000, 500,
    1, 0, 0, 0, 8000, 400, 40, 2, 100, 2500,
    2400, 900, 810000, 1, 1, 0, 200, 200, 0, 0,
    0, 0.05, 2400, 2500, 100, 0, 0, 0, 0, 0,
    0, 200, 500000, 10, 1000, 65535, 512, 50, 20,
    100, 50, 200, 50, 2000000, 1000000, 4000000, 500000
]}

# extreme values (should always trigger)
EXTREME_PAYLOAD = {"features": [999999] * 10}

# minimal payload (tests padding logic)
MINIMAL_PAYLOAD = {"features": [80, 443, 6, 100]}


def send_request(payload):
    """Send a single predict request and return the latency and result."""
    start = time.perf_counter()
    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        elapsed = (time.perf_counter() - start) * 1000  # milliseconds
        if response.status_code == 200:
            result = response.json()
            return {
                "latency_ms": elapsed,
                "score": result["malicious_score"],
                "threat": result["threat_detected"],
                "success": True,
            }
        else:
            return {"latency_ms": elapsed, "success": False, "error": response.status_code}
    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        return {"latency_ms": elapsed, "success": False, "error": str(e)}


def measure_latency(payload, label, num_requests=100):
    """Send multiple requests sequentially and report latency statistics."""
    print(f"\n--- Latency Test: {label} ({num_requests} requests) ---")

    # warm up the API with 5 throwaway requests
    for _ in range(5):
        send_request(payload)

    latencies = []
    scores = []
    threats = 0
    failures = 0

    for i in range(num_requests):
        result = send_request(payload)
        if result["success"]:
            latencies.append(result["latency_ms"])
            scores.append(result["score"])
            if result["threat"]:
                threats += 1
        else:
            failures += 1

    if not latencies:
        print(f"  ALL {num_requests} REQUESTS FAILED")
        return None

    stats = {
        "label": label,
        "requests": num_requests,
        "successes": len(latencies),
        "failures": failures,
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "mean_ms": statistics.mean(latencies),
        "median_ms": statistics.median(latencies),
        "p95_ms": sorted(latencies)[int(len(latencies) * 0.95)],
        "p99_ms": sorted(latencies)[int(len(latencies) * 0.99)],
        "stdev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
        "threat_rate": threats / len(latencies),
        "avg_score": statistics.mean(scores),
    }

    print(f"  Successes: {stats['successes']}/{num_requests}")
    print(f"  Min:    {stats['min_ms']:.2f} ms")
    print(f"  Mean:   {stats['mean_ms']:.2f} ms")
    print(f"  Median: {stats['median_ms']:.2f} ms")
    print(f"  P95:    {stats['p95_ms']:.2f} ms")
    print(f"  P99:    {stats['p99_ms']:.2f} ms")
    print(f"  Max:    {stats['max_ms']:.2f} ms")
    print(f"  StdDev: {stats['stdev_ms']:.2f} ms")
    print(f"  Threat rate: {stats['threat_rate']*100:.1f}%")
    print(f"  Avg score: {stats['avg_score']:.6f}")

    return stats


def stress_test(payload, label, num_requests=500, max_workers=10):
    """Send concurrent requests to measure throughput under load."""
    print(f"\n--- Stress Test: {label} ({num_requests} requests, {max_workers} concurrent) ---")

    # warm up
    for _ in range(5):
        send_request(payload)

    latencies = []
    failures = 0

    start_time = time.perf_counter()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(send_request, payload) for _ in range(num_requests)]
        for future in as_completed(futures):
            result = future.result()
            if result["success"]:
                latencies.append(result["latency_ms"])
            else:
                failures += 1

    total_time = time.perf_counter() - start_time

    if not latencies:
        print(f"  ALL {num_requests} REQUESTS FAILED")
        return None

    throughput = len(latencies) / total_time

    stats = {
        "label": label,
        "requests": num_requests,
        "concurrency": max_workers,
        "successes": len(latencies),
        "failures": failures,
        "total_time_s": total_time,
        "throughput_rps": throughput,
        "min_ms": min(latencies),
        "mean_ms": statistics.mean(latencies),
        "median_ms": statistics.median(latencies),
        "p95_ms": sorted(latencies)[int(len(latencies) * 0.95)],
        "p99_ms": sorted(latencies)[int(len(latencies) * 0.99)],
        "max_ms": max(latencies),
    }

    print(f"  Successes: {stats['successes']}/{num_requests} ({failures} failures)")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Throughput: {throughput:.1f} requests/sec")
    print(f"  Mean latency: {stats['mean_ms']:.2f} ms")
    print(f"  Median latency: {stats['median_ms']:.2f} ms")
    print(f"  P95 latency: {stats['p95_ms']:.2f} ms")
    print(f"  P99 latency: {stats['p99_ms']:.2f} ms")
    print(f"  Max latency: {stats['max_ms']:.2f} ms")

    return stats


def detection_accuracy_test():
    """Test detection accuracy across different traffic profiles."""
    print("\n--- Detection Accuracy Test ---")

    test_cases = [
        ("Benign (minimal)", MINIMAL_PAYLOAD, False),
        ("Benign (full 77-feature)", BENIGN_FULL, False),
        ("DDoS attack profile", DDOS_PAYLOAD, True),
        ("Data exfiltration profile", EXFIL_PAYLOAD, True),
        ("Extreme values", EXTREME_PAYLOAD, True),
    ]

    results = []
    correct = 0
    total = len(test_cases)

    for label, payload, expected_threat in test_cases:
        result = send_request(payload)
        if result["success"]:
            actual_threat = result["threat"]
            match = actual_threat == expected_threat
            if match:
                correct += 1
            status = "PASS" if match else "FAIL"
            print(f"  [{status}] {label}: score={result['score']:.6f} "
                  f"threat={actual_threat} (expected={expected_threat})")
            results.append({
                "label": label,
                "score": result["score"],
                "expected": expected_threat,
                "actual": actual_threat,
                "correct": match,
            })
        else:
            print(f"  [ERROR] {label}: request failed")
            results.append({"label": label, "correct": False, "error": True})

    print(f"\n  Accuracy: {correct}/{total} ({correct/total*100:.0f}%)")
    return results


def run_full_evaluation():
    """Run all evaluation benchmarks and save results."""
    print("=" * 60)
    print("  AI-NGFW Performance Evaluation")
    print(f"  {datetime.now().isoformat()}")
    print("=" * 60)

    # check API is reachable
    print("\n[CHECK] verifying API is reachable...")
    try:
        result = send_request(MINIMAL_PAYLOAD)
        if not result["success"]:
            print(f"[ERROR] API returned error: {result.get('error')}")
            print("Make sure the API container is running.")
            sys.exit(1)
        print(f"[CHECK] API is responding. Score: {result['score']:.6f}")
    except Exception as e:
        print(f"[ERROR] cannot reach API: {e}")
        sys.exit(1)

    all_results = {}

    # 1. detection accuracy
    print("\n" + "=" * 60)
    print("  TEST 1: Detection Accuracy")
    print("=" * 60)
    all_results["accuracy"] = detection_accuracy_test()

    # 2. latency measurements (sequential, per-request timing)
    print("\n" + "=" * 60)
    print("  TEST 2: Inference Latency (Sequential)")
    print("=" * 60)
    all_results["latency_benign"] = measure_latency(BENIGN_FULL, "Benign traffic", 100)
    all_results["latency_ddos"] = measure_latency(DDOS_PAYLOAD, "DDoS traffic", 100)
    all_results["latency_exfil"] = measure_latency(EXFIL_PAYLOAD, "Exfiltration traffic", 100)
    all_results["latency_minimal"] = measure_latency(MINIMAL_PAYLOAD, "Minimal payload (4 features)", 100)

    # 3. stress test (concurrent requests, throughput measurement)
    print("\n" + "=" * 60)
    print("  TEST 3: Stress Test (Concurrent Load)")
    print("=" * 60)
    all_results["stress_5"] = stress_test(BENIGN_FULL, "5 concurrent", 200, 5)
    all_results["stress_10"] = stress_test(BENIGN_FULL, "10 concurrent", 500, 10)
    all_results["stress_20"] = stress_test(BENIGN_FULL, "20 concurrent", 500, 20)
    all_results["stress_50"] = stress_test(BENIGN_FULL, "50 concurrent", 500, 50)

    # 4. summary
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)

    # latency summary table
    print("\nLatency Summary (sequential, 100 requests each):")
    print(f"  {'Profile':<30} {'Mean':>8} {'Median':>8} {'P95':>8} {'P99':>8}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for key in ["latency_benign", "latency_ddos", "latency_exfil", "latency_minimal"]:
        s = all_results[key]
        if s:
            print(f"  {s['label']:<30} {s['mean_ms']:>7.1f}ms {s['median_ms']:>7.1f}ms "
                  f"{s['p95_ms']:>7.1f}ms {s['p99_ms']:>7.1f}ms")

    # throughput summary table
    print("\nThroughput Summary (concurrent requests):")
    print(f"  {'Concurrency':<30} {'RPS':>10} {'Mean Lat':>10} {'P95 Lat':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")
    for key in ["stress_5", "stress_10", "stress_20", "stress_50"]:
        s = all_results[key]
        if s:
            print(f"  {s['label']:<30} {s['throughput_rps']:>9.1f} "
                  f"{s['mean_ms']:>9.1f}ms {s['p95_ms']:>9.1f}ms")

    # save results to file
    output_path = "/app/zeek_logs/evaluation_results.txt"
    try:
        with open(output_path, "w") as f:
            f.write(f"AI-NGFW Performance Evaluation Results\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"{'=' * 60}\n\n")

            f.write("Detection Accuracy:\n")
            for r in all_results.get("accuracy", []):
                if "error" not in r:
                    status = "PASS" if r["correct"] else "FAIL"
                    f.write(f"  [{status}] {r['label']}: score={r['score']:.6f} "
                            f"expected={r['expected']} actual={r['actual']}\n")
            f.write("\n")

            f.write("Latency (sequential, ms):\n")
            f.write(f"  {'Profile':<30} {'Mean':>8} {'Median':>8} {'P95':>8} {'P99':>8}\n")
            for key in ["latency_benign", "latency_ddos", "latency_exfil", "latency_minimal"]:
                s = all_results[key]
                if s:
                    f.write(f"  {s['label']:<30} {s['mean_ms']:>7.1f}ms {s['median_ms']:>7.1f}ms "
                            f"{s['p95_ms']:>7.1f}ms {s['p99_ms']:>7.1f}ms\n")
            f.write("\n")

            f.write("Throughput (concurrent):\n")
            f.write(f"  {'Concurrency':<30} {'RPS':>10} {'Mean Lat':>10} {'P95 Lat':>10}\n")
            for key in ["stress_5", "stress_10", "stress_20", "stress_50"]:
                s = all_results[key]
                if s:
                    f.write(f"  {s['label']:<30} {s['throughput_rps']:>9.1f} "
                            f"{s['mean_ms']:>9.1f}ms {s['p95_ms']:>9.1f}ms\n")

        print(f"\nResults saved to {output_path}")
    except Exception as e:
        print(f"\nCould not save results file: {e}")
        print("Results were printed above.")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    run_full_evaluation()