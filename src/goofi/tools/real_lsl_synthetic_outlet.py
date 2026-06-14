"""Launch the SyntheticLSLEEG node as a real pylsl StreamOutlet."""

import argparse
import signal
from typing import Optional

from goofi.nodes.inputs.syntheticlsleeg import SyntheticLSLEEG


def build_node(args: argparse.Namespace) -> SyntheticLSLEEG:
    node = SyntheticLSLEEG.create_standalone()
    node.params.lsl.stream_name.value = args.stream_name
    node.params.lsl.source_name.value = args.source_id
    node.params.lsl.stream_type.value = args.stream_type
    node.params.signal.sampling_rate.value = args.sampling_frequency
    node.params.signal.chunk_size.value = args.chunk_size
    node.params.signal.seed.value = args.seed
    node.params.signal.noise.value = args.noise
    node.params.signal.amplitude.value = args.amplitude
    node.params.signal.channel_labels.value = ",".join(SyntheticLSLEEG.DEFAULT_CHANNELS)
    node.setup()
    return node


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Publish deterministic 64-channel synthetic EEG over real LSL.")
    parser.add_argument("--stream-name", default="goofi_windows_real_lsl_eeg")
    parser.add_argument("--source-id", default="goofi_windows_real_lsl_source")
    parser.add_argument("--stream-type", default="EEG")
    parser.add_argument("--sampling-frequency", type=int, default=250)
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--noise", type=float, default=0.0, help="Use 0 for exact deterministic validation.")
    parser.add_argument("--amplitude", type=float, default=20.0)
    parser.add_argument("--max-chunks", type=int, default=0, help="Stop after this many chunks; 0 runs until interrupted.")
    args = parser.parse_args(argv)

    stop = False

    def _stop(_signum, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    node = build_node(args)
    print(
        f"Publishing {len(node.channel_labels)} {args.stream_type} channels as name={args.stream_name!r} "
        f"source_id={args.source_id!r} sfreq={args.sampling_frequency}Hz chunk_size={args.chunk_size}",
        flush=True,
    )
    chunks = 0
    try:
        while not stop and (args.max_chunks <= 0 or chunks < args.max_chunks):
            node.process()
            chunks += 1
    finally:
        node.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
