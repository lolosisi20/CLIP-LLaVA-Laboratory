import subprocess
import sys
import tempfile

TIMEOUT_SECONDS = 60
OUTPUT_LIMIT = 4000


def run_python(code: str) -> str:
    """Runs code in a fresh, isolated interpreter in an empty folder and returns what it printed.
    It is a separate process with a time limit, not a security sandbox: only model-written analysis code goes here."""
    with tempfile.TemporaryDirectory(prefix="review-python-") as folder:
        try:
            finished = subprocess.run([sys.executable, "-I", "-c", code], cwd=folder, capture_output=True,
                                      text=True, timeout=TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            return f"Stopped after {TIMEOUT_SECONDS} s."
    output = finished.stdout + (f"\n[stderr]\n{finished.stderr}" if finished.stderr else "")
    return output[-OUTPUT_LIMIT:] or "(printed nothing: use print())"
