import importlib.util
import subprocess
import sys
import time

import pytest

from goofi.tools.real_lsl_validate import main as validate_main

pytestmark = pytest.mark.real_lsl


def _pylsl_available():
    if importlib.util.find_spec("pylsl") is None:
        return False, "pylsl is not installed"
    try:
        import pylsl  # noqa: F401
    except Exception as exc:
        return False, f"pylsl/liblsl is unavailable: {exc}"
    return True, ""


@pytest.mark.skipif(not _pylsl_available()[0], reason=_pylsl_available()[1])
def test_real_lsl_synthetic_outlet_validates_with_lslclient():
    stream_name = "goofi_pytest_real_lsl_eeg"
    source_id = "goofi_pytest_real_lsl_source"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "goofi.tools.real_lsl_synthetic_outlet",
            "--stream-name",
            stream_name,
            "--source-id",
            source_id,
            "--sampling-frequency",
            "250",
            "--chunk-size",
            "10",
            "--noise",
            "0",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        time.sleep(1.0)
        assert proc.poll() is None, proc.stdout.read() if proc.stdout else "outlet exited"
        assert (
            validate_main(
                [
                    "--stream-name",
                    stream_name,
                    "--source-id",
                    source_id,
                    "--sampling-frequency",
                    "250",
                    "--chunk-size",
                    "10",
                    "--samples",
                    "40",
                    "--timeout",
                    "8",
                ]
            )
            == 0
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
