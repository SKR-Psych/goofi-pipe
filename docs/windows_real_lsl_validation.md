# Windows real LSL validation

This workflow validates real operating-system-level Lab Streaming Layer (LSL) transport for the deterministic 64-channel `SyntheticLSLEEG` node and the production `LSLClient`. It does not use the fake or in-memory test transport, and it does not claim clinical-grade latency.

## Recommended environment

Use Python 3.11 on Windows. The project supports Python 3.9 through 3.12, but Python 3.11 is a conservative choice for `pylsl`/`liblsl` compatibility.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

## Install and verify pylsl/liblsl

```powershell
python -m pip install pylsl
python - <<'PY'
import pylsl
print('pylsl import ok')
print(pylsl.library_info())
PY
```

Recent `pylsl` wheels often include or locate the native `liblsl` runtime automatically. If import fails with a DLL/native-library error, install a matching `liblsl` runtime and ensure the DLL directory is on `PATH` before starting Python.

## Windows Firewall

LSL discovery uses network discovery even for local workflows. Allow Python through Windows Defender Firewall for private networks when prompted. If discovery fails, check VPNs, restrictive firewalls, and security software that blocks multicast/broadcast traffic.

## Launch the synthetic outlet

Terminal 1:

```powershell
goofi-real-lsl-outlet --stream-name goofi_windows_real_lsl_eeg --source-id goofi_windows_real_lsl_source --sampling-frequency 250 --chunk-size 25 --noise 0
```

Expected startup output includes the stream name, source ID, 64 channels, stream type `EEG`, sampling frequency, and chunk size.

## Launch receiver validation

Terminal 2:

```powershell
goofi-real-lsl-validate --stream-name goofi_windows_real_lsl_eeg --source-id goofi_windows_real_lsl_source --sampling-frequency 250 --chunk-size 25 --samples 250 --json-report real_lsl_report.json
```

Expected successful output starts with:

```text
Real LSL validation passed
```

The summary reports source LSL timestamp, local reception timestamp, estimated transport latency, chunk interval, continuity status, missing and duplicate counts, timestamp regressions, total samples, effective sampling rate, and reconnect status. Treat these as measured values only.

## Run tests

Fake/in-memory unit tests remain separate from real-LSL integration tests:

```powershell
pytest tests/test_syntheticlsleeg.py
```

Real LSL tests are marked `real_lsl` and skip clearly if `pylsl` or native `liblsl` is unavailable:

```powershell
pytest -m real_lsl tests/test_real_lsl_transport.py -ra
```

## Troubleshooting stream discovery

- Confirm the outlet terminal is still running.
- Verify stream identifiers match exactly: stream name, source ID, and type `EEG`.
- Re-run the validation with a longer `--timeout`.
- Allow Python through Windows Firewall on private networks.
- Disable VPNs or network filters temporarily.
- Stop stale outlet processes that may publish duplicate stream identifiers.

## Stop stale processes cleanly

Press `Ctrl+C` in the outlet terminal. If a process is detached, use Task Manager or PowerShell:

```powershell
Get-Process python | Where-Object {$_.Path -like '*python*'}
Stop-Process -Id <PID>
```

Use care not to stop unrelated Python work.
