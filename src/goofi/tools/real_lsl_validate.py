"""Validate a real LSL stream using goofi's LSLClient implementation."""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

from goofi.nodes.inputs.lslclient import LSLClient
from goofi.nodes.inputs.syntheticlsleeg import SyntheticLSLEEG


def _labels_from_info(info):
    ch = info.desc().child("channels").child("channel")
    labels = []
    for _ in range(info.channel_count()):
        labels.append(ch.child_value("label"))
        ch = ch.next_sibling()
    return labels


def _connect(stream_name, source_id, stream_type, timeout):
    client = LSLClient.create_standalone()
    client.params.lsl_stream.source_name.value = source_id
    client.params.lsl_stream.stream_name.value = stream_name
    client.params.lsl_stream.source_type.value = stream_type
    client.setup()
    deadline = time.monotonic() + timeout
    while client.client is None and time.monotonic() < deadline:
        client.available_streams = None
        client.connect()
        time.sleep(0.2)
    if client.client is None:
        raise TimeoutError(
            f"Timed out discovering LSL stream name={stream_name!r}, source_id={source_id!r}, type={stream_type!r}"
        )
    return client


def validate(args: argparse.Namespace) -> dict:
    client = _connect(args.stream_name, args.source_id, args.stream_type, args.timeout)
    info = client.client.info()
    labels = _labels_from_info(info)
    assert info.name() == args.stream_name
    assert info.source_id() == args.source_id
    assert info.type() == args.stream_type
    assert info.channel_count() == 64
    assert labels == SyntheticLSLEEG.DEFAULT_CHANNELS
    assert info.nominal_srate() == args.sampling_frequency

    expected_node = SyntheticLSLEEG.create_standalone()
    expected_node.channel_labels = SyntheticLSLEEG.DEFAULT_CHANNELS
    expected_node.params.signal.noise.value = args.noise
    expected_node.params.signal.amplitude.value = args.amplitude
    expected_node.rng = np.random.default_rng(args.seed)

    chunks = []
    timestamps = []
    local_times = []
    total = 0
    deadline = time.monotonic() + args.timeout
    while total < args.samples and time.monotonic() < deadline:
        samples, ts = client.client.pull_chunk(timeout=0.5, max_samples=args.chunk_size)
        if samples:
            arr = np.asarray(samples, dtype=np.float32).T
            assert arr.shape[0] == 64
            chunks.append(arr)
            timestamps.extend(ts)
            local_times.append(client.pylsl.local_clock())
            total += arr.shape[1]
    if total < args.samples:
        raise TimeoutError(f"Received only {total}/{args.samples} samples")

    data = np.concatenate(chunks, axis=1)[:, : args.samples]
    timestamps = np.asarray(timestamps[: args.samples], dtype=float)
    assert data.shape == (64, args.samples)
    assert len(chunks) >= 2
    assert np.all(np.diff(timestamps) >= 0)

    # The outlet may have started before the receiver connected. Infer the deterministic sample offset
    # rather than assuming the first received sample is stream sample zero.
    search = expected_node._make_chunk(0, args.search_samples, args.sampling_frequency).astype(np.float32)
    probe = min(args.chunk_size, args.samples)
    offset = None
    for candidate in range(0, args.search_samples - args.samples + 1):
        if np.allclose(data[:, :probe], search[:, candidate : candidate + probe], rtol=args.rtol, atol=args.atol):
            offset = candidate
            break
    if offset is None:
        raise AssertionError(
            f"Could not match received deterministic samples within first {args.search_samples} generated samples"
        )
    expected = search[:, offset : offset + args.samples]
    np.testing.assert_allclose(data, expected, rtol=args.rtol, atol=args.atol)

    diffs = np.diff(timestamps)
    regressions = int(np.sum(diffs < 0))
    expected_dt = 1.0 / args.sampling_frequency
    missing = int(np.sum(diffs > expected_dt * 1.5))
    duplicates = int(np.sum(diffs == 0))
    effective_rate = float((len(timestamps) - 1) / (timestamps[-1] - timestamps[0])) if len(timestamps) > 1 else 0.0
    latency = [lt - float(ts) for lt, ts in zip(local_times, timestamps[:: args.chunk_size])]

    client.disconnect()
    reconnected = _connect(args.stream_name, args.source_id, args.stream_type, args.timeout)
    reconnected.disconnect()

    report = {
        "stream_name": args.stream_name,
        "source_id": args.source_id,
        "stream_type": args.stream_type,
        "source_lsl_timestamp": float(timestamps[-1]),
        "local_reception_timestamp": float(local_times[-1]),
        "estimated_transport_latency_seconds": float(np.median(latency)) if latency else None,
        "chunk_interval_seconds": float(np.median(np.diff(local_times))) if len(local_times) > 1 else None,
        "sample_continuity": missing == 0 and duplicates == 0 and regressions == 0,
        "missing_sample_count": missing,
        "duplicate_sample_count": duplicates,
        "timestamp_regressions": regressions,
        "total_received_samples": int(total),
        "effective_sampling_rate_hz": effective_rate,
        "first_matched_sample_index": int(offset),
        "reconnect_ok": True,
    }
    return report


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate real OS-level LSL transport with LSLClient.")
    parser.add_argument("--stream-name", default="goofi_windows_real_lsl_eeg")
    parser.add_argument("--source-id", default="goofi_windows_real_lsl_source")
    parser.add_argument("--stream-type", default="EEG")
    parser.add_argument("--sampling-frequency", type=int, default=250)
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--search-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--amplitude", type=float, default=20.0)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--json-report", type=Path)
    args = parser.parse_args(argv)
    report = validate(args)
    print("Real LSL validation passed")
    for key, value in report.items():
        print(f"{key}: {value}")
    if args.json_report:
        args.json_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
