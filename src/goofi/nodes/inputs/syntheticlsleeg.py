import math
import time
from typing import List

import numpy as np

from goofi.node import Node
from goofi.params import FloatParam, IntParam, StringParam


class SyntheticLSLEEG(Node):
    """
    Publishes a deterministic synthetic 64-channel EEG stream over Lab Streaming Layer (LSL).

    This node is intended for end-to-end pipeline testing with LSLClient. It creates an LSL outlet with EEG-like
    sine-wave activity, low-amplitude Gaussian noise, standard 64-channel actiCAP-style labels, and a configurable
    source ID / stream name so downstream LSLClient nodes can connect to it exactly like a live amplifier stream.

    Inputs:
    - None.

    Outputs:
    - None. Data is published externally as an LSL stream.
    """

    DEFAULT_CHANNELS = [
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

    def config_params():
        return {
            "lsl": {
                "source_name": "goofi_synthetic_64ch",
                "stream_name": "synthetic_eeg_64ch",
                "stream_type": "EEG",
            },
            "signal": {
                "sampling_rate": IntParam(250, 1, 5000, doc="Synthetic stream sampling rate in Hz."),
                "chunk_size": IntParam(25, 1, 1000, doc="Samples pushed per LSL chunk."),
                "amplitude": FloatParam(20.0, 0.0, 1000.0, doc="Base sine amplitude in arbitrary EEG units."),
                "noise": FloatParam(1.0, 0.0, 1000.0, doc="Gaussian noise standard deviation."),
                "seed": IntParam(13, 0, 2**31 - 1, doc="Random seed for reproducible noise."),
                "channel_labels": StringParam(
                    ",".join(SyntheticLSLEEG.DEFAULT_CHANNELS),
                    doc="Comma-separated channel labels. Defaults to a 64-channel actiCAP-style montage.",
                ),
            },
            "common": {"autotrigger": True, "max_frequency": 60.0},
        }

    def setup(self):
        import pylsl

        self.pylsl = pylsl
        self.sample_index = 0
        self.rng = np.random.default_rng(self.params.signal.seed.value)
        self.channel_labels = self._parse_channel_labels(self.params.signal.channel_labels.value)
        self._create_outlet()

    def process(self):
        sfreq = self.params.signal.sampling_rate.value
        chunk_size = self.params.signal.chunk_size.value
        chunk = self._make_chunk(self.sample_index, chunk_size, sfreq)
        self.outlet.push_chunk(chunk.T.astype(np.float32).tolist())
        self.sample_index += chunk_size
        time.sleep(chunk_size / sfreq)
        return None

    def terminate(self):
        self.outlet = None

    def _create_outlet(self):
        sfreq = self.params.signal.sampling_rate.value
        self.info = self.pylsl.StreamInfo(
            self.params.lsl.stream_name.value,
            self.params.lsl.stream_type.value,
            len(self.channel_labels),
            sfreq,
            "float32",
            self.params.lsl.source_name.value,
        )
        channels = self.info.desc().append_child("channels")
        for label in self.channel_labels:
            channel = channels.append_child("channel")
            channel.append_child_value("label", label)
            channel.append_child_value("type", "EEG")
            channel.append_child_value("unit", "microvolts")
        self.outlet = self.pylsl.StreamOutlet(self.info)

    def _make_chunk(self, start_sample: int, chunk_size: int, sfreq: float) -> np.ndarray:
        t = (np.arange(start_sample, start_sample + chunk_size) / sfreq)[None, :]
        channel_idx = np.arange(len(self.channel_labels))[:, None]
        alpha = np.sin(2 * math.pi * (8.0 + channel_idx % 5) * t)
        beta = 0.35 * np.sin(2 * math.pi * (18.0 + channel_idx % 7) * t)
        slow = 0.2 * np.sin(2 * math.pi * (2.0 + channel_idx % 3) * t)
        noise = self.rng.normal(0.0, self.params.signal.noise.value, size=(len(self.channel_labels), chunk_size))
        return self.params.signal.amplitude.value * (alpha + beta + slow) + noise

    @staticmethod
    def _parse_channel_labels(labels: str) -> List[str]:
        parsed = [label.strip() for label in labels.split(",") if label.strip()]
        if not parsed:
            raise ValueError("At least one channel label is required.")
        return parsed

    def lsl_source_name_changed(self, _):
        self._create_outlet()

    def lsl_stream_name_changed(self, _):
        self._create_outlet()

    def lsl_stream_type_changed(self, _):
        self._create_outlet()

    def signal_sampling_rate_changed(self, _):
        self._create_outlet()

    def signal_seed_changed(self, _):
        self.rng = np.random.default_rng(self.params.signal.seed.value)

    def signal_channel_labels_changed(self, _):
        self.channel_labels = self._parse_channel_labels(self.params.signal.channel_labels.value)
        self._create_outlet()
