# Synthetic 64-channel EEG LSL pipeline

This fork includes `SyntheticLSLEEG`, a deterministic 64-channel EEG-like LSL publisher, plus validation coverage for:

```text
SyntheticLSLEEG -> LSLClient -> Filter -> Buffer -> PSD -> PowerBandEEG
```

`SyntheticLSLEEG` publishes sample-major LSL chunks. `LSLClient` emits channel-major arrays shaped `(64, n_samples)` for downstream EEG feature extraction.

## Validated data contract

- `LSLClient` can preserve sample timestamps in `meta["timestamps"]`.
- For timestamp-aware chunks, `LSLClient` also stores timestamps on `meta["channels"]["dim1"]`, matching goofi-pipe's axis metadata convention.
- Strict 64-channel EEG validation is opt-in through `validation.strict_64ch_eeg`.
- Strict validation checks exactly 64 channels, expected sampling frequency, duplicate labels, missing labels, extra labels, and exact label order.
- The expected label order is the `SyntheticLSLEEG.DEFAULT_CHANNELS` montage:

```text
Fp1, Fp2, F7, F3, Fz, F4, F8, FC5, FC1, FC2, FC6, T7, C3, Cz, C4, T8,
CP5, CP1, CP2, CP6, P7, P3, Pz, P4, P8, PO9, O1, Oz, O2, PO10, AF7,
AF3, AF4, AF8, F5, F1, F2, F6, FT7, FC3, FC4, FT8, C5, C1, C2, C6,
TP7, CP3, CPz, CP4, TP8, P5, P1, P2, P6, PO7, PO3, POz, PO4, PO8,
Iz, FCz, AFz, Fpz
```

## Suggested node settings

### SyntheticLSLEEG

- `lsl.source_name`: `goofi_synthetic_64ch`
- `lsl.stream_name`: `synthetic_eeg_64ch`
- `lsl.stream_type`: `EEG`
- `signal.sampling_rate`: `250`
- `signal.chunk_size`: `25`

### LSLClient

- `lsl_stream.source_name`: `goofi_synthetic_64ch`
- `lsl_stream.stream_name`: `synthetic_eeg_64ch`
- `lsl_stream.source_type`: `EEG`
- `validation.strict_64ch_eeg`: enabled for the synthetic 64-channel validation pipeline
- `validation.expected_channel_count`: `64`
- `validation.expected_sfreq`: `250`
- `validation.expected_channel_labels`: the comma-separated label list above

Strict validation is disabled by default so existing non-EEG and non-64-channel LSL streams keep working.

## Numerical validation

The tests cover:

- deterministic synthetic `(64, n_samples)` signal generation;
- `LSLClient` timestamp preservation and strict channel validation;
- causal `Filter` continuity across multiple chunks;
- rolling `Buffer` correctness with variable chunk sizes;
- `PSD` detection for known theta, alpha, and beta signals;
- `PowerBandEEG` alpha-band dominance for a known 10 Hz signal;
- the documented `SyntheticLSLEEG -> Filter -> Buffer -> PSD -> PowerBandEEG` feature pipeline.

## Running validation tests

From the goofi-pipe repository root:

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q tests/test_syntheticlsleeg.py tests/test_lslclient_64ch_validation.py tests/test_eeg_feature_pipeline.py
python -m black --check src tests
python -m compileall -q src tests
```

The `LSLClient` validation tests use a fake LSL inlet to exercise the real `LSLClient.process()` path without requiring native LSL discovery. Real OS-level `pylsl` discovery and transport are intentionally left for the later Windows real-LSL validation phase.

## Remaining limitations

- Real `pylsl` discovery is not validated in this phase.
- Brain Products hardware-specific assumptions are not included.
- Model inference is not included.
- Artefact rejection is not included.
- Unity and Firebase integration are not included.
- `Classifier -> Table -> OSCOut` remains documented as a downstream integration pattern but is not validated here.
