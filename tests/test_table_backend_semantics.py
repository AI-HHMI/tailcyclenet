"""The pandas-free contract of the table-only modules (the polars migration's one live guard)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_target_modules_do_not_load_pandas_in_a_fresh_subprocess():
    """Table-only modules must not import pandas in a fresh interpreter."""
    for module in ("tailcyclenet.infer.bridge", "tailcyclenet.scorer.qc"):
        code = f"import sys; import {module}; print('pandas' in sys.modules)"
        result = subprocess.run([sys.executable, "-c", code],
                                cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "False", module
