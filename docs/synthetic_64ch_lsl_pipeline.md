# Synthetic 64-channel EEG LSL pipeline

This repository now includes `SyntheticLSLEEG`, an input node that publishes a deterministic 64-channel EEG-like LSL stream for end-to-end testing with `LSLClient`.

Recommended graph:

```text
SyntheticLSLEEG
→ LSLClient
→ Filter
→ Buffer
→ PSD
→ PowerBandEEG
→ Classifier
→ Table
→ OSCOut
```

`Table` is required between `Classifier` and `OSCOut` because `Classifier.probs` is an ARRAY output, while `OSCOut` accepts TABLE input and serializes table entries into OSC addresses.

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

### Filter

- Enable bandpass for the EEG range you need, for example 1-45 Hz.
- Enable notch at 50 or 60 Hz according to your lab mains frequency.
- Use causal filtering for true real-time behavior.

### Buffer

- Use seconds as the unit for model windows, for example 1-2 seconds.
- Buffer on the sample axis (`axis = -1`) so 64 channels are preserved on `dim0`.

### PSD and PowerBandEEG

- Configure PSD over the same sample axis (`axis = -1`).
- `PowerBandEEG` produces delta, theta, alpha, low beta, high beta, and gamma arrays.

### Classifier

- Feed one selected feature array or a joined/reduced feature vector into `Classifier.data`.
- Train inside goofi-pipe for prototyping, or replace with a custom exported-model inference node for production.

### OSCOut

- Use a `Table` node to place classifier probabilities under a stable key such as `probs`.
- Set `OSCOut.osc.prefix` to a Unity-specific prefix, for example `/eeg`.
- Unity will receive probabilities at `/eeg/probs`.
