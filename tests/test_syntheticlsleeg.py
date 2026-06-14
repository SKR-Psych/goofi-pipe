import numpy as np

from goofi.nodes.inputs.syntheticlsleeg import SyntheticLSLEEG


def test_default_channel_labels_are_64_unique():
    labels = SyntheticLSLEEG.DEFAULT_CHANNELS
    assert len(labels) == 64
    assert len(set(labels)) == 64
    assert labels[:5] == ["Fp1", "Fp2", "F7", "F3", "Fz"]


def test_parse_channel_labels_rejects_empty():
    try:
        SyntheticLSLEEG._parse_channel_labels(" , , ")
    except ValueError as exc:
        assert "At least one channel label" in str(exc)
    else:
        raise AssertionError("Empty channel label strings should be rejected.")


def test_make_chunk_shape_and_determinism():
    node_a = SyntheticLSLEEG.create_standalone()
    node_a.channel_labels = SyntheticLSLEEG.DEFAULT_CHANNELS
    node_a.rng = np.random.default_rng(123)
    node_a.params.signal.noise.value = 0.0
    node_a.params.signal.amplitude.value = 1.0

    node_b = SyntheticLSLEEG.create_standalone()
    node_b.channel_labels = SyntheticLSLEEG.DEFAULT_CHANNELS
    node_b.rng = np.random.default_rng(123)
    node_b.params.signal.noise.value = 0.0
    node_b.params.signal.amplitude.value = 1.0

    chunk_a = node_a._make_chunk(start_sample=0, chunk_size=25, sfreq=250)
    chunk_b = node_b._make_chunk(start_sample=0, chunk_size=25, sfreq=250)

    assert chunk_a.shape == (64, 25)
    np.testing.assert_allclose(chunk_a, chunk_b)
