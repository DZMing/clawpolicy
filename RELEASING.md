# Releasing

## Prerequisites

- CI is green on all supported platforms.
- `python3 -m pytest tests/ -v` passes.
- `python3 scripts/check_docs_consistency.py` passes.
- `CHANGELOG.md` updated.

## Release Steps

1. Bump version in `pyproject.toml` and changelog.
2. Create release branch/tag:
   - `git tag vX.Y.Z`
   - `git push origin vX.Y.Z`
3. Build package:
   - `python3 -m build`
4. Validate artifacts:
   - `python3 -m twine check dist/*`
5. Publish:
   - TestPyPI first, then PyPI.
6. Create GitHub Release and paste changelog summary.

## Post-release

- Monitor issues for 7 days.
- Prepare `X.Y.(Z+1)` hotfix if needed.

## ClawHub publish sanity

Treat GitHub as the canonical release source and ClawHub as a derivative distribution surface.

Before or after publishing to ClawHub:

1. Verify `LICENSE`, `pyproject.toml`, and README agree.
2. Run `python3 scripts/check_clawhub_consistency.py --repo-root . --slug clawpolicy` when possible.
3. If the current ClawHub CLI still rejects normal publish/sync because of `acceptLicenseTerms`, use `node scripts/publish_clawhub_manual.mjs --skill-dir <dir> --slug clawpolicy --version <X.Y.Z> --name ClawPolicy`.
4. Confirm GitHub tag/release exists for the canonical version before treating the distribution as final.
