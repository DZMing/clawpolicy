# Code Review Guidelines

## Project Overview

ClawPolicy is a Python policy engine for autonomous agent execution. It manages policy lifecycles (`hint -> candidate -> confirmed -> suspended -> archived`) with CLI and Python API.

## Review Priorities

### Critical (must block PR)

- Security: policy bypass, unauthorized privilege escalation, unsafe deserialization
- Data integrity: policy state corruption, incorrect lifecycle transitions
- Breaking changes: public API signature changes without version bump

### High (should fix before merge)

- Test coverage: new code paths without corresponding tests in `tests/`
- Error handling: bare `except`, swallowed exceptions, missing error context
- CLI safety: destructive commands without confirmation prompts

### Medium (suggest improvements)

- Type hints missing on public functions
- Docstrings missing on public API
- Code duplication across modules

## Testing Requirements

- All PRs must pass: `python3 -m pytest tests/`
- New features require at least one test case
- Policy lifecycle changes require integration tests

## File Structure

- `clawpolicy/` — core library code
- `tests/` — test suite
- `scripts/` — utility scripts
- `templates/` — policy templates
- `docs/` — documentation

## Style

- Python 3.8+ compatible
- Follow existing patterns in the codebase
- Use `pyproject.toml` for project configuration
