import numpy as np
import pytest

from goofi.data import Data, DataType
from goofi.nodes.inputs.lslclient import LSLClient
from goofi.nodes.inputs.syntheticlsleeg import SyntheticLSLEEG


def test_lslclient_preserves_64_channel_data_labels_sfreq_and_timestamps():
    source = _synthetic_source()
    expected = source._make_chunk(start_sample=0, chunk_size=32, sfreq=250)
    timestamps = source._make_timestamps(start_sample=0, chunk_size=32, sfreq=250)

    client = _strict_lslclient(
        labels=SyntheticLSLEEG.DEFAULT_CHANNELS,
        samples=expected.T.astype(np.float32),
        timestamps=timestamps,
        sfreq=250,
    )

    result = client.process(None, None, None)
    samples, meta = result["out"]

    assert samples.shape == (64, 32)
    assert meta["sfreq"] == 250
    assert meta["channels"]["dim0"] == SyntheticLSLEEG.DEFAULT_CHANNELS
    np.testing.assert_allclose(meta["timestamps"], timestamps)
    np.testing.assert_allclose(meta["channels"]["dim1"], timestamps)
    np.testing.assert_allclose(samples, expected.astype(np.float32), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        (SyntheticLSLEEG.DEFAULT_CHANNELS[:-1], "Expected exactly 64 EEG channels"),
        (
            SyntheticLSLEEG.DEFAULT_CHANNELS[:10]
            + [SyntheticLSLEEG.DEFAULT_CHANNELS[9]]
            + SyntheticLSLEEG.DEFAULT_CHANNELS[11:],
            "Duplicate EEG channel labels",
        ),
        (
            SyntheticLSLEEG.DEFAULT_CHANNELS[:-1] + ["NotARealChannel"],
            "EEG channel labels differ",
        ),
        (
            [SyntheticLSLEEG.DEFAULT_CHANNELS[1], SyntheticLSLEEG.DEFAULT_CHANNELS[0]] + SyntheticLSLEEG.DEFAULT_CHANNELS[2:],
            "EEG channel order mismatch at index 0",
        ),
        (SyntheticLSLEEG.DEFAULT_CHANNELS + ["EXTRA"], "Expected exactly 64 EEG channels"),
    ],
)
def test_lslclient_channel_integrity_failures_are_clear(labels, message):
    samples = np.zeros((12, len(labels)), dtype=np.float32)
    timestamps = np.arange(samples.shape[0], dtype=float) / 250
    client = _strict_lslclient(labels=labels, samples=samples, timestamps=timestamps)

    with pytest.raises(ValueError, match=message):
        client.process(None, None, None)


def test_lslclient_sampling_frequency_validation_is_clear():
    samples = np.zeros((12, 64), dtype=np.float32)
    timestamps = np.arange(samples.shape[0], dtype=float) / 500
    client = _strict_lslclient(
        labels=SyntheticLSLEEG.DEFAULT_CHANNELS,
        samples=samples,
        timestamps=timestamps,
        sfreq=500,
    )

    with pytest.raises(ValueError, match="Expected sampling frequency 250.0 Hz"):
        client.process(None, None, None)


def _synthetic_source():
    source = SyntheticLSLEEG.create_standalone()
    source.channel_labels = SyntheticLSLEEG.DEFAULT_CHANNELS
    source.rng = np.random.default_rng(123)
    source.params.signal.noise.value = 0.0
    source.params.signal.amplitude.value = 1.0
    return source


def _strict_lslclient(labels, samples, timestamps, sfreq=250):
    client = LSLClient.create_standalone()
    client.client = _FakeInlet(labels=labels, samples=samples, timestamps=timestamps, sfreq=sfreq)
    client.params.validation.strict_64ch_eeg.value = True
    client.params.validation.expected_sfreq.value = 250.0
    client.params.lsl_stream.source_name.value = "goofi_synthetic_64ch"
    client.params.lsl_stream.stream_name.value = "synthetic_eeg_64ch"
    client.params.lsl_stream.source_type.value = "EEG"
    return client


class _FakeInlet:
    def __init__(self, labels, samples, timestamps, sfreq=250, stream_type="EEG"):
        self._info = _FakeInfo(labels, sfreq=sfreq, stream_type=stream_type)
        self._samples = samples.tolist()
        self._timestamps = timestamps.tolist()

    def pull_chunk(self):
        return self._samples, self._timestamps

    def info(self):
        return self._info


class _FakeInfo:
    def __init__(self, labels, sfreq, stream_type):
        self.labels = labels
        self.sfreq = sfreq
        self.stream_type = stream_type

    def desc(self):
        return _FakeDesc(self.labels)

    def type(self):
        return self.stream_type

    def channel_count(self):
        return len(self.labels)

    def nominal_srate(self):
        return self.sfreq


class _FakeDesc:
    def __init__(self, labels):
        self.labels = labels

    def child(self, name):
        assert name == "channels"
        return self

    def child_value(self, _):
        return ""

    def next_sibling(self):
        return self

    def child(self, name):  # noqa: F811
        if name == "channels":
            return self
        if name == "channel":
            return _FakeChannelCursor(self.labels)
        raise KeyError(name)


class _FakeChannelCursor:
    def __init__(self, labels, index=0):
        self.labels = labels
        self.index = index

    def child_value(self, name):
        assert name == "label"
        if self.index >= len(self.labels):
            return ""
        return self.labels[self.index]

    def next_sibling(self):
        return _FakeChannelCursor(self.labels, self.index + 1)


def _data(array, sfreq=250):
    timestamps = np.arange(array.shape[-1], dtype=float) / sfreq
    return Data(
        DataType.ARRAY,
        np.asarray(array, dtype=float),
        {"sfreq": sfreq, "channels": {"dim1": timestamps.tolist()}, "timestamps": timestamps.tolist()},
    )
