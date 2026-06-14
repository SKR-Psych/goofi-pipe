import socket
import time
from threading import Thread, current_thread
from typing import Any, Dict, List, Tuple

import numpy as np
from tabulate import tabulate

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, FloatParam, IntParam, StringParam

# pylsl default wait is 1.0s; short waits can miss outlets that advertise slightly later.
# Multiple passes merged by uid fixes Muse-style multi-stream devices (same source_id + name, different type).
LSL_RESOLVE_ROUNDS = 2
LSL_RESOLVE_WAIT_S = 2.0
LSL_RESOLVE_PAUSE_S = 0.25


def _stream_info_key(info) -> Tuple:
    """Stable key for deduplicating pylsl.StreamInfo across resolve passes."""
    u = info.uid()
    if u:
        return ("uid", u)
    return (
        "fallback",
        info.source_id(),
        info.name(),
        info.type(),
        info.hostname(),
        info.channel_count(),
        info.nominal_srate(),
    )


class LSLClient(Node):
    """
    This node connects to a Lab Streaming Layer (LSL) stream and receives real-time data from it. It discovers available LSL streams on the network, connects to the specified source and stream, reads chunks of incoming data, and outputs this data along with relevant metadata such as channel names and sampling frequency. The node is suitable for live signal acquisition from any source that publishes data via LSL.

    Inputs:
    - source_name: The LSL source ID to connect to.
    - stream_name: The LSL stream name within the specified source.
    - source_type: The LSL stream type string (StreamInfo.type(), e.g. EEG or GYRO). Optional when empty; set when multiple streams share the same source ID and stream name.

    Outputs:
    - out: The acquired data as an array, along with metadata including sampling frequency and channel names.
    """

    def config_params():
        return {
            "lsl_stream": {
                "source_name": "goofi",
                "stream_name": "",
                "source_type": "",
                "refresh": BoolParam(False, trigger=True),
            },
            "validation": {
                "strict_64ch_eeg": BoolParam(
                    False,
                    doc=(
                        "Validate incoming EEG streams against the expected 64-channel montage. "
                        "Disabled by default so existing arbitrary LSL streams keep working."
                    ),
                ),
                "expected_channel_count": IntParam(64, 1, 512),
                "expected_sfreq": FloatParam(250.0, 0.0, 10000.0),
                "expected_channel_labels": StringParam(
                    ",".join(
                        [
                            "Fp1",
                            "Fp2",
                            "F7",
                            "F3",
                            "Fz",
                            "F4",
                            "F8",
                            "FC5",
                            "FC1",
                            "FC2",
                            "FC6",
                            "T7",
                            "C3",
                            "Cz",
                            "C4",
                            "T8",
                            "CP5",
                            "CP1",
                            "CP2",
                            "CP6",
                            "P7",
                            "P3",
                            "Pz",
                            "P4",
                            "P8",
                            "PO9",
                            "O1",
                            "Oz",
                            "O2",
                            "PO10",
                            "AF7",
                            "AF3",
                            "AF4",
                            "AF8",
                            "F5",
                            "F1",
                            "F2",
                            "F6",
                            "FT7",
                            "FC3",
                            "FC4",
                            "FT8",
                            "C5",
                            "C1",
                            "C2",
                            "C6",
                            "TP7",
                            "CP3",
                            "CPz",
                            "CP4",
                            "TP8",
                            "P5",
                            "P1",
                            "P2",
                            "P6",
                            "PO7",
                            "PO3",
                            "POz",
                            "PO4",
                            "PO8",
                            "Iz",
                            "FCz",
                            "AFz",
                            "Fpz",
                        ]
                    ),
                    doc="Comma-separated expected EEG channel labels in stream order.",
                ),
            },
            "common": {"autotrigger": True},
        }

    def config_input_slots():
        return {
            "source_name": DataType.STRING,
            "stream_name": DataType.STRING,
            "source_type": DataType.STRING,
        }

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def setup(self):
        """Initialize and start the LSL client."""
        import pylsl

        self.pylsl = pylsl

        if hasattr(self, "client"):
            self.disconnect()

        self.client = None
        self.lsl_discover_thread = None
        self.ch_names = None

        self.available_streams = None

        # initialize list of streams
        self.connect()

    def process(self, source_name: Data, stream_name: Data, source_type: Data) -> Dict[str, Tuple[np.ndarray, Dict[str, Any]]]:
        """Fetch the next chunk of data from the client."""
        if source_name is not None:
            self.params.lsl_stream.source_name.value = source_name.data
            self.lsl_stream_source_name_changed(source_name.data)
            self.input_slots["source_name"].clear()
        if stream_name is not None:
            self.params.lsl_stream.stream_name.value = stream_name.data
            self.lsl_stream_stream_name_changed(stream_name.data)
            self.input_slots["stream_name"].clear()
        if source_type is not None:
            self.params.lsl_stream.source_type.value = source_type.data
            self.lsl_stream_source_type_changed(source_type.data)
            self.input_slots["source_type"].clear()

        if self.client is None:
            if not self.connect():
                return None

        try:
            # fetch data
            samples, timestamps = self.client.pull_chunk()
        except Exception as e:
            print(f"Error fetching data from LSL stream: {e}")
            self.setup()
            return

        samples = np.array(samples).T

        if timestamps is None or len(timestamps) != samples.shape[-1]:
            timestamps = None

        if samples.size == 0:
            return

        try:
            ch_names = self._get_channel_names()
            self.ch_names = ch_names
        except Exception as e:
            print(f"Error fetching channel names from LSL stream: {e}")
            self.setup()
            return

        self._validate_stream(ch_names, samples)

        meta = {
            "sfreq": self.client.info().nominal_srate(),
            "channels": {"dim0": self.ch_names},
        }
        if timestamps is not None:
            timestamps = [float(ts) for ts in timestamps]
            meta["timestamps"] = timestamps
            meta["channels"]["dim1"] = timestamps
        return {"out": (samples, meta)}

    def _get_channel_names(self) -> List[str]:
        ch_info = self.client.info().desc().child("channels").child("channel")
        ch_type = self.client.info().type().lower()
        ch_names = []
        for k in range(1, self.client.info().channel_count() + 1):
            ch_names.append(ch_info.child_value("label") or "{} {:03d}".format(ch_type.upper(), k))
            ch_info = ch_info.next_sibling()
        return ch_names

    def _validate_stream(self, ch_names: List[str], samples: np.ndarray) -> None:
        if not self.params.validation.strict_64ch_eeg.value:
            return

        expected_count = self.params.validation.expected_channel_count.value
        expected_sfreq = self.params.validation.expected_sfreq.value
        expected_labels = [
            label.strip() for label in self.params.validation.expected_channel_labels.value.split(",") if label.strip()
        ]

        info = self.client.info()
        if info.type() != "EEG":
            raise ValueError(f'Expected EEG stream type, got "{info.type()}".')
        if info.channel_count() != expected_count or samples.shape[0] != expected_count:
            raise ValueError(
                f"Expected exactly {expected_count} EEG channels, got "
                f"stream={info.channel_count()} and samples={samples.shape[0]}."
            )
        if not np.isclose(info.nominal_srate(), expected_sfreq):
            raise ValueError(f"Expected sampling frequency {expected_sfreq} Hz, got {info.nominal_srate()} Hz.")
        duplicate_labels = sorted({label for label in ch_names if ch_names.count(label) > 1})
        if duplicate_labels:
            raise ValueError(f"Duplicate EEG channel labels present: {duplicate_labels}.")
        missing_labels = [label for label in expected_labels if label not in ch_names]
        extra_labels = [label for label in ch_names if label not in expected_labels]
        if missing_labels or extra_labels:
            raise ValueError(f"EEG channel labels differ; missing={missing_labels}, extra={extra_labels}.")
        if ch_names != expected_labels:
            for index, (expected, actual) in enumerate(zip(expected_labels, ch_names)):
                if expected != actual:
                    raise ValueError(f"EEG channel order mismatch at index {index}: " f"expected {expected!r}, got {actual!r}.")

    def connect(self) -> bool:
        """Connect to the LSL stream."""
        if self.client is not None:
            self.disconnect()
        if self.available_streams is None:
            self.lsl_stream_refresh_changed(True)

        # find the stream
        source_name = self.params.lsl_stream.source_name.value
        stream_name = self.params.lsl_stream.stream_name.value
        source_type = self.params.lsl_stream.source_type.value

        matches = {}
        for info in self.available_streams:
            h, s, n = info.hostname(), info.source_id(), info.name()
            t = info.type()
            if s != source_name:
                continue
            if len(stream_name) > 0 and n != stream_name:
                continue
            if len(source_type) > 0 and t != source_type:
                continue
            key = (s, n, t)
            if key in matches and h == socket.gethostname():
                # prefer local streams
                matches[key] = info
            elif key not in matches:
                # otherwise, prefer the first match
                matches[key] = info

        if len(matches) != 1:
            if self.lsl_discover_thread is None:
                # check if new streams arrived
                self.lsl_discover_thread = Thread(
                    target=self.lsl_stream_refresh_changed, args=(True,), daemon=True, name="lsl_discover_thread"
                )
                self.lsl_discover_thread.start()

                type_suffix = f', type "{source_type}"' if source_type else ""
                if len(matches) == 0:
                    print(f'\nCould not find source "{source_name}" with stream "{stream_name}"{type_suffix}.')
                else:
                    # ms = tabulate(
                    #     [list(m) for m in matches],
                    #     headers=["Source ID", "Stream Name"],
                    #     tablefmt="simple_outline",
                    # )
                    # print(f'\nFound multiple streams matching source="{source_name}", name="{stream_name}":\n{ms}.')
                    print(
                        f'\nFound multiple streams matching source="{source_name}", name="{stream_name}"{type_suffix}:\n{matches}.'
                    )
            return False

        # if len(matches) != 1:
        #     print(f'\nFound multiple streams matching source="{source_name}", name="{stream_name}":\n{matches}.')
        #     return False

        # connect to the stream
        self.client = self.pylsl.StreamInlet(info=list(matches.values())[0], recover=True)
        return True

    def disconnect(self) -> None:
        """Disconnect from the LSL stream."""
        if self.client is not None:
            try:
                self.client.close_stream()
            except:
                pass
            self.client = None

    def _resolve_stream_infos(self) -> List:
        """Resolve LSL outlets; merge several passes so all streams are discovered (see pylsl.resolve_streams)."""
        merged: Dict[Tuple, Any] = {}
        for round_i in range(LSL_RESOLVE_ROUNDS):
            for info in self.pylsl.resolve_streams(wait_time=LSL_RESOLVE_WAIT_S):
                merged[_stream_info_key(info)] = info
            if round_i + 1 < LSL_RESOLVE_ROUNDS:
                time.sleep(LSL_RESOLVE_PAUSE_S)
        return list(merged.values())

    def lsl_stream_refresh_changed(self, value: bool) -> None:
        self.available_streams = self._resolve_stream_infos()
        stream_data = sorted(
            [[info.source_id(), info.name(), info.type(), info.hostname()] for info in self.available_streams],
            key=lambda x: x[0],
        )

        # print("\nAvailable LSL streams:")
        # print(tabulate(stream_data, headers=["Source ID", "Stream Name", "Host Name"], tablefmt="simple_outline"))
        # print()
        print("\nAvailable LSL streams:")
        print(f"{'Source ID':<36} {'Stream Name':<25} {'Type':<12} {'Host Name':<20}")
        print("-" * 98)

        for source_id, stream_name, stream_type, host_name in stream_data:
            print(f"{source_id:<36} {stream_name:<25} {stream_type:<12} {host_name:<20}")
        print()

        if current_thread().name == "lsl_discover_thread":
            self.lsl_discover_thread = None

    def lsl_stream_source_name_changed(self, value: str) -> None:
        try:
            if self.client is not None and value != self.client.info().source_id():
                self.setup()
        except:
            # stream might have been lost
            self.setup()

    def lsl_stream_stream_name_changed(self, value: str) -> None:
        try:
            if self.client is not None and value != self.client.info().name():
                self.setup()
        except:
            # stream might have been lost
            self.setup()

    def lsl_stream_source_type_changed(self, value: str) -> None:
        try:
            if self.client is not None and value != self.client.info().type():
                self.setup()
        except:
            # stream might have been lost
            self.setup()
