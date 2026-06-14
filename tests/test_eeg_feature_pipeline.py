import numpy as np

from goofi.data import Data, DataType
from goofi.nodes.analysis.powerbandeeg import PowerBandEEG
from goofi.nodes.inputs.syntheticlsleeg import SyntheticLSLEEG
from goofi.nodes.signal.buffer import Buffer
from goofi.nodes.signal.filter import Filter
from goofi.nodes.signal.psd import PSD


def test_causal_filter_is_continuous_across_chunks():
    message = _sine_message([2.0, 10.0], sfreq=250, seconds=2.0)

    full_filter = Filter.create_standalone()
    full_filter.setup()
    full_filter.params.bandpass.apply.value = True
    full_filter.params.bandpass.f_low.value = 1.0
    full_filter.params.bandpass.f_high.value = 45.0
    full_filter.params.bandpass.method.value = "Causal"
    full = full_filter.process(message)["filtered_data"][0]

    chunked_filter = Filter.create_standalone()
    chunked_filter.setup()
    chunked_filter.params.bandpass.apply.value = True
    chunked_filter.params.bandpass.f_low.value = 1.0
    chunked_filter.params.bandpass.f_high.value = 45.0
    chunked_filter.params.bandpass.method.value = "Causal"
    first = Data(DataType.ARRAY, message.data[:, :137], _slice_meta(message.meta, 0, 137))
    second = Data(DataType.ARRAY, message.data[:, 137:], _slice_meta(message.meta, 137, None))
    split = np.concatenate(
        [
            chunked_filter.process(first)["filtered_data"][0],
            chunked_filter.process(second)["filtered_data"][0],
        ],
        axis=-1,
    )

    np.testing.assert_allclose(split, full, rtol=1e-12, atol=1e-12)


def test_rolling_buffer_handles_variable_chunk_sizes_and_timestamp_axis():
    buffer = Buffer.create_standalone()
    buffer.setup()
    buffer.params.buffer.size.value = 10
    buffer.params.buffer.axis.value = -1
    buffer.params.buffer.unit.value = "samples"

    chunks = [
        np.arange(0, 3, dtype=float)[None, :],
        np.arange(3, 11, dtype=float)[None, :],
        np.arange(11, 15, dtype=float)[None, :],
    ]

    output = None
    offset = 0
    for chunk in chunks:
        output = buffer.process(_data(chunk, sfreq=10, start_sample=offset), None)
        offset += chunk.shape[-1]

    samples, meta = output["out"]
    assert samples.shape == (1, 10)
    np.testing.assert_array_equal(samples[0], np.arange(5, 15, dtype=float))
    np.testing.assert_allclose(meta["channels"]["dim1"], np.arange(5, 15, dtype=float) / 10)


def test_psd_detects_known_theta_alpha_and_beta_signals():
    psd_node = PSD.create_standalone()
    psd_node.setup()
    psd_node.params.psd.method.value = "fft"
    psd_node.params.psd.f_min.value = 1
    psd_node.params.psd.f_max.value = 30
    psd = psd_node.process(_sine_message([6.0, 10.0, 18.0], sfreq=256, seconds=2.0))["psd"]
    power, meta = psd
    freqs = np.asarray(meta["channels"]["dim1"])
    peak_freqs = freqs[np.argmax(power, axis=-1)]

    np.testing.assert_allclose(peak_freqs, [6.0, 10.0, 18.0], atol=0.25)


def test_powerbandeeg_identifies_expected_dominant_alpha_band():
    psd_node = PSD.create_standalone()
    psd_node.setup()
    psd_node.params.psd.method.value = "fft"
    psd_node.params.psd.f_min.value = 1
    psd_node.params.psd.f_max.value = 30
    power, meta = psd_node.process(_sine_message([10.0, 10.0, 10.0], sfreq=256, seconds=2.0))["psd"]

    band_node = PowerBandEEG.create_standalone()
    bands = band_node.process(Data(DataType.ARRAY, power, meta))

    band_power = {band: values[0] for band, values in bands.items()}
    dominant = max(band_power, key=lambda band: float(np.mean(band_power[band])))

    assert dominant == "alpha"
    assert np.all(band_power["alpha"] > band_power["theta"])
    assert np.all(band_power["alpha"] > band_power["lowbeta"])


def test_documented_synthetic_pipeline_to_powerbandeeg_features():
    source = SyntheticLSLEEG.create_standalone()
    source.channel_labels = SyntheticLSLEEG.DEFAULT_CHANNELS
    source.rng = np.random.default_rng(123)
    source.params.signal.noise.value = 0.0
    source.params.signal.amplitude.value = 1.0
    chunk = source._make_chunk(0, 250, 250)

    filt = Filter.create_standalone()
    filt.setup()
    buffer = Buffer.create_standalone()
    buffer.setup()
    buffer.params.buffer.size.value = 250
    buffer.params.buffer.axis.value = -1
    buffer.params.buffer.unit.value = "samples"
    psd_node = PSD.create_standalone()
    psd_node.setup()
    psd_node.params.psd.method.value = "fft"
    psd_node.params.psd.f_min.value = 1
    psd_node.params.psd.f_max.value = 45
    band_node = PowerBandEEG.create_standalone()

    filtered = filt.process(_data(chunk, sfreq=250))["filtered_data"]
    buffered = buffer.process(Data(DataType.ARRAY, filtered[0], filtered[1]), None)["out"]
    psd = psd_node.process(Data(DataType.ARRAY, buffered[0], buffered[1]))["psd"]
    bands = band_node.process(Data(DataType.ARRAY, psd[0], psd[1]))

    band_power = {band: values[0] for band, values in bands.items()}
    dominant = max(band_power, key=lambda band: float(np.mean(band_power[band])))

    assert buffered[0].shape == (64, 250)
    assert dominant == "alpha"


def _sine_message(freqs, sfreq, seconds):
    t = np.arange(0, seconds, 1.0 / sfreq)
    data = np.vstack([np.sin(2.0 * np.pi * freq * t) for freq in freqs])
    return _data(data, sfreq=sfreq)


def _slice_meta(meta, start, stop):
    sliced = {
        "sfreq": meta["sfreq"],
        "channels": {"dim1": meta["channels"]["dim1"][start:stop]},
        "timestamps": meta["timestamps"][start:stop],
    }
    return sliced


def _data(array, sfreq=250, start_sample=0):
    timestamps = np.arange(start_sample, start_sample + array.shape[-1], dtype=float) / sfreq
    return Data(
        DataType.ARRAY,
        np.asarray(array, dtype=float),
        {"sfreq": sfreq, "channels": {"dim1": timestamps.tolist()}, "timestamps": timestamps.tolist()},
    )
