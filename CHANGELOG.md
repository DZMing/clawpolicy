# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and this project follows Semantic Versioning.

## [3.0.2] - 2026-03-11

### Fixed

- Realigned repository metadata, release docs, and public package wording with the canonical ClawPolicy positioning: an explainable autonomous execution policy engine for low-touch, auditable agent execution.
- Prepared the GitHub source of truth to match the corrected ClawHub 3.0.2 publish payload and install/verification instructions.

### Changed

- Bumped the release line from 3.0.1 to 3.0.2 across package metadata, README surfaces, docs headings, and embedded version strings so GitHub and distribution metadata stay in sync.

## [3.0.1] - 2026-03-08

### Fixed

- Tightened the public Python surface around `clawpolicy` so docs and examples no longer point at `lib` as a supported import path.
- Unified policy-state defaults so local workflows consistently center on project-local `.clawpolicy/` behavior.
- Stopped read-only status flows from implicitly bootstrapping policy directories.
- Expanded docs consistency coverage so public API and storage-path drift are caught before release.
- Updated repository metadata and docs links to the canonical `DZMing/clawpolicy` GitHub home.

### Changed

- Released a patch follow-up to the 3.0.0 rename cutover so the published package, docs, and repository metadata stay aligned.

## [3.0.0] - 2026-03-08

### Breaking Changes

- Project renamed to ClawPolicy across distribution metadata, import entrypoint, CLI command, and release documentation.
- Canonical local storage is `.clawpolicy/policy/` with `rules.json`, `playbooks.json`, and `policy_events.jsonl`.
- Operational supervision is centered on `clawpolicy policy ...` and `python -m clawpolicy`.

### Changed

- Refactored the runtime around the policy lifecycle `hint -> candidate -> confirmed -> suspended -> archived`.
- Standardized the public CLI surface around the single supported command `clawpolicy`.
- Removed deprecated naming, deprecated bootstrap paths, and deprecated release text from the repository.

### Added

- Stable public API surface for policy storage and confirmation integrations:
  - `ConfirmationAPI`
  - `create_api`
  - `Rule`, `Playbook`, `PolicyEvent`
  - `PolicyStore`
  - `MarkdownToPolicyConverter`
  - `PolicyToMarkdownExporter`

- Added scheduled dependency audit workflow (`pip-audit`).

### Breaking Changes

- None.

### Known Limitations

- Phase3 full dependency installation is heavier (includes optional ML/monitoring stack).
- Legacy ClawPolicy config/model paths are still supported for backward compatibility.
