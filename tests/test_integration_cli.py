#!/usr/bin/env python3
"""
integration CLI Basic usability testing
"""

import subprocess
import sys
from pathlib import Path


def test_integration_module_help():
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-m", "lib.integration", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage:" in result.stdout.lower()


def test_public_cli_module_help():
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-m", "clawpolicy", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage:" in result.stdout.lower()


def test_public_cli_module_policy_status(tmp_path: Path):
    result = subprocess.run(
        [sys.executable, "-m", "clawpolicy", "policy", "status"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Policy lifecycle status" in result.stdout
