#!/usr/bin/env python3
"""
Document consistency check script testing
"""

import subprocess
import sys
from pathlib import Path


def test_docs_consistency_check_script_passes():
    repo_root = Path(__file__).resolve().parent.parent
    script = repo_root / "scripts" / "check_docs_consistency.py"

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False
    )

    assert result.returncode == 0, result.stdout + result.stderr
