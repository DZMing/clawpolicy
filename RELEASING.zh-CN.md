# Releasing Guide (Legacy zh-CN Path)

This file path is kept for backward compatibility.

Use the primary English release runbook:

- RELEASING.md

## ClawHub 发布检查

把 GitHub 视为 canonical release source，把 ClawHub 视为派生的分发/展示面。

发布到 ClawHub 前后，至少执行：

1. 核对 `LICENSE`、`pyproject.toml`、README 是否一致。
2. 条件允许时执行 `python3 scripts/check_clawhub_consistency.py --repo-root . --slug clawpolicy`。
3. 若当前 ClawHub CLI 因 `acceptLicenseTerms` 导致 `publish/sync` 失效，则使用 `node scripts/publish_clawhub_manual.mjs --skill-dir <dir> --slug clawpolicy --version <X.Y.Z> --name ClawPolicy`。
4. GitHub tag/release 存在后，才把这次分发视为最终完成。
