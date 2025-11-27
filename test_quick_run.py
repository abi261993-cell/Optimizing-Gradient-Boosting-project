import subprocess
import sys
import os
import pytest

SCRIPT = os.path.join(os.path.dirname(_file_), "..", "pure_numpy_gbm.py")

def test_quick_run():
    assert os.path.exists(SCRIPT)
    res = subprocess.run([sys.executable, SCRIPT, "--quick"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    stdout = res.stdout.decode('utf-8', errors='ignore')
    stderr = res.stderr.decode('utf-8', errors='ignore')
    # must exit 0
    assert res.returncode == 0, f"Script failed. stderr:\n{stderr}"
    # check summary text in stdout
    assert "Comparison Summary" in stdout or "Test RMSE" in stdout
