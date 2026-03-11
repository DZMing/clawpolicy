# License Source of Truth

## Canonical Source

The **canonical source of truth** for ClawPolicy code, licensing, and repository metadata is:

- GitHub repository: `https://github.com/DZMing/clawpolicy`
- Repository license file: `LICENSE`
- Packaging metadata: `pyproject.toml`

For ClawPolicy, the canonical code license is **MIT**.

## ClawHub Listing Relationship

ClawHub is treated as a **distribution and discovery surface**, not the canonical legal source for repository licensing.

If ClawHub metadata differs from the GitHub repository (for example a historical listing showing `MIT-0`), the GitHub repository remains authoritative for:

- source code provenance
- license interpretation for the repository code
- release metadata tied to tagged repository commits

## Why This Exists

Historically, the ClawHub skill listing for `clawpolicy` drifted from the repository truth in both summary and license-adjacent metadata. This file exists so future publish, review, and consistency checks have an explicit rule:

```yaml
canonical_source:
  code: GitHub
  repository_license: GitHub
  distribution_listing: ClawHub
```

## Operator Rule

Before any future ClawHub publish or release cut:

1. Verify `LICENSE`, `pyproject.toml`, and README still agree.
2. Verify the tagged GitHub release is the canonical release source.
3. Treat ClawHub metadata as a published derivative that must be checked against GitHub, not the other way around.
