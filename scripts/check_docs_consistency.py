#!/usr/bin/env python3
"""
Check whether README key metrics match the actual code state.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.contracts import ACTION_VECTOR_DIM  # noqa: E402

# Public API exports that should be available from clawpolicy package
EXPECTED_PUBLIC_API = {
    "ConfirmationAPI",
    "PolicyEvent",
    "PolicyStore",
    "Playbook",
    "Rule",
    "MarkdownToPolicyConverter",
    "PolicyToMarkdownExporter",
    "create_api",
    "__version__",
}


def _extract_first_int(pattern: str, content: str) -> Optional[int]:
    match = re.search(pattern, content)
    if not match:
        return None
    return int(match.group(1))


def collect_pytest_count(repo_root: Path) -> int:
    """Collect real test count via pytest."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "--collect-only", "-q"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout + result.stderr
    if result.returncode != 0:
        raise RuntimeError(f"pytest collect failed: {output}")

    count = _extract_first_int(r"(\d+)\s+tests\s+collected", output)
    if count is not None:
        return count

    # Under some configs pytest prints per-file counts:
    # tests/test_x.py: 5
    line_counts = re.findall(r"^tests/.+:\s*(\d+)\s*$", output, flags=re.MULTILINE)
    if line_counts:
        return sum(int(item) for item in line_counts)

    raise RuntimeError(f"cannot parse pytest collected count: {output}")


def validate_readme_metrics(repo_root: Path, expected_tests: int) -> list[str]:
    errors: list[str] = []

    en_content = (repo_root / "README.md").read_text(encoding="utf-8")
    zh_path = repo_root / "README.zh-CN.md"
    zh_content = zh_path.read_text(encoding="utf-8") if zh_path.exists() else ""

    # Legacy README.zh-CN.md may be translated to English-only content and
    # omit metric blocks entirely. Treat zh fields as optional compatibility checks.
    zh_tests = _extract_first_int(r"\*\*Total Tests\*\*:\s*(\d+)", zh_content)
    if zh_tests is None:
        zh_tests = _extract_first_int(r"\*\*Total number of tests\*\*:\s*(\d+)", zh_content)
    en_tests = _extract_first_int(r"\*\*Total Tests\*\*:\s*(\d+)", en_content)
    zh_action_dim = _extract_first_int(
        r"`Action`:\s*Action data class \((\d+) dimensions\)",
        zh_content,
    )
    if zh_action_dim is None:
        zh_action_dim = _extract_first_int(r"Action:\s*action data class.*?(\d+)", zh_content)
    en_action_dim = _extract_first_int(
        r"`Action`:\s*Action data class \((\d+) dimensions\)",
        en_content,
    )

    if zh_tests is not None and zh_tests != expected_tests:
        errors.append(f"README.zh-CN.md Total Tests={zh_tests}, expected {expected_tests}.")

    if en_tests is None:
        errors.append("README.md is missing the 'Total Tests' field.")
    elif en_tests != expected_tests:
        errors.append(f"README.md Total Tests={en_tests}, expected {expected_tests}.")

    if zh_action_dim is not None and zh_action_dim != ACTION_VECTOR_DIM:
        errors.append(f"README.zh-CN.md Action dimension={zh_action_dim}, expected {ACTION_VECTOR_DIM}.")

    if en_action_dim is None:
        errors.append("README.md is missing the action dimension field.")
    elif en_action_dim != ACTION_VECTOR_DIM:
        errors.append(f"README.md Action dimensions={en_action_dim}, expected {ACTION_VECTOR_DIM}.")

    if "<repository_url>" in en_content:
        errors.append("README.md still contains the <repository_url> placeholder.")

    required_files = ["requirements.txt", "requirements-full.txt", "requirements-dev.txt"]
    for rel in required_files:
        if not (repo_root / rel).exists():
            errors.append(f"Missing dependency file: {rel}")

    governance_files = [
        "CHANGELOG.md",
        "CONTRIBUTING.md",
        "CONTRIBUTING.zh-CN.md",
        "SECURITY.md",
        "SECURITY.zh-CN.md",
        "SUPPORT.md",
        "SUPPORT.zh-CN.md",
        "CODE_OF_CONDUCT.md",
        "CODE_OF_CONDUCT.zh-CN.md",
        "RELEASING.md",
        "RELEASING.zh-CN.md",
    ]
    for rel in governance_files:
        if not (repo_root / rel).exists():
            errors.append(f"Missing governance document: {rel}")

    manifest_content = (repo_root / "MANIFEST.in").read_text(encoding="utf-8")
    for rel in governance_files:
        if f"include {rel}" not in manifest_content:
            errors.append(f"MANIFEST.in does not include governance doc: {rel}")

    security_en = (repo_root / "SECURITY.md").read_text(encoding="utf-8")
    security_zh = (repo_root / "SECURITY.zh-CN.md").read_text(encoding="utf-8")
    advisory_url = "https://github.com/DZMing/clawpolicy/security/advisories/new"
    if advisory_url not in security_en:
        errors.append("SECURITY.md is missing the GitHub private advisory link.")
    if advisory_url not in security_zh:
        errors.append("SECURITY.zh-CN.md is missing the GitHub private advisory link.")

    # Check: README should use `from clawpolicy import`, not `from lib import`
    if "from lib import" in en_content:
        errors.append(
            "README.md uses 'from lib import' but should use 'from clawpolicy import'. "
            "lib is internal implementation, clawpolicy is the public package."
        )
    if "from lib import" in zh_content:
        errors.append(
            "README.zh-CN.md uses 'from lib import' but should use 'from clawpolicy import'. "
            "lib is internal implementation, clawpolicy is the public package."
        )

    # Check: README should mention clawpolicy as the public package
    if "from clawpolicy import" not in en_content:
        errors.append(
            "README.md doesn't show 'from clawpolicy import' example. "
            "This is the stable public API surface."
        )

    # Check: clawpolicy/__init__.py should export expected public API
    try:
        import clawpolicy
        actual_exports = set(clawpolicy.__all__)
        missing_exports = EXPECTED_PUBLIC_API - actual_exports
        if missing_exports:
            errors.append(
                f"clawpolicy/__init__.py missing exports: {missing_exports}. "
                f"Expected: {EXPECTED_PUBLIC_API}, got: {actual_exports}"
            )
    except Exception as e:
        errors.append(f"Failed to import clawpolicy package: {e}")

    # Check: Default state directory should be .clawpolicy/ consistently
    state_dir_mentions = re.findall(r"`\.clawpolicy`|\.clawpolicy/|\.clawpolicy\"", en_content)
    if not state_dir_mentions:
        errors.append(
            "README.md doesn't mention '.clawpolicy/' as the canonical state directory. "
            "This is the 3.0.1 default for local policy storage."
        )

    return errors


def main() -> int:
    repo_root = REPO_ROOT

    try:
        total_tests = collect_pytest_count(repo_root)
        errors = validate_readme_metrics(repo_root, total_tests)
    except Exception as exc:  # pragma: no cover - CLI fallback
        print(f"[docs-check] ERROR: {exc}")
        return 1

    if errors:
        print("[docs-check] FAILED")
        for err in errors:
            print(f"- {err}")
        return 1

    print(f"[docs-check] OK: tests={total_tests}, action_dim={ACTION_VECTOR_DIM}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
