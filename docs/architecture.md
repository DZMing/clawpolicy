# Architecture

## 3.0.1 Core Flow (Policy-First)

1. **Init**: `clawpolicy init` creates `.clawpolicy/policy/` with canonical storage
2. **Import**: `MarkdownToPolicyConverter` migrates USER/SOUL/AGENTS.md → policy assets
3. **Decision**: `ConfirmationAPI` evaluates tasks against policy rules
4. **Feedback**: Execution outcomes update rule confidence via `IntelligentConfirmation`
5. **Promotion**: Successful hints graduate through `hint → candidate → confirmed`
6. **Demotion**: Failed rules get `suspended` or `archived`

### Policy Lifecycle

```
hint (weak)
  ↓ (evidence积累)
candidate (待验证)
  ↓ (success_streak ≥ N)
confirmed (自动执行)
  ↓ (连续失败)
suspended (暂停)
  ↓ (手动或时间)
archived (归档)
```

## Module Layers

### Policy Core (Primary 3.0.1 Surface)

- `policy_models.py`: canonical `Rule`, `Playbook`, `PolicyEvent`
- `policy_store.py`: canonical policy asset persistence
- `policy_resolution.py`: scope inference and precedence resolution
- `confirmation.py`: runtime truth loop, event recording, feedback application
- `promotion.py`: `candidate → confirmed` promotion gates
- `demotion.py`: suspension, reactivation, and archive gates
- `api.py`: stable confirmation API surface (`ConfirmationAPI`, `create_api`)
- `cli.py`: initialization, status, supervision, export, and inspection commands

### Markdown Integration

- `md_to_policy.py`: `MarkdownToPolicyConverter` (USER/SOUL/AGENTS.md → policy)
- `policy_to_md.py`: `PolicyToMarkdownExporter` (policy → Markdown export)

### Optional Phase 3 (RL/Training)

- `environment.py`: RL environment with State/Action contracts
- `reward.py`: multi-objective reward computation
- `agent.py`: RL policy learning and action selection
- `learner.py`: preference derivation and RL optimization
- `distributed_trainer.py`, `hyperparameter_tuner.py`, etc.: advanced training

## Public API

```python
from clawpolicy import (
    ConfirmationAPI,      # Runtime confirmation decisions
    PolicyStore,          # Policy asset persistence
    Rule, Playbook, PolicyEvent,  # Policy models
    MarkdownToPolicyConverter,   # MD → Policy
    PolicyToMarkdownExporter,    # Policy → MD
    create_api,           # Convenience factory
)
```

**Note**: `lib` is internal implementation. Use `clawpolicy` for external integrations.

## State Storage

- **Default location**: `.clawpolicy/policy/` in project root
- **Canonical files**:
  - `rules.json`: Rule registry with confidence scores
  - `playbooks.json`: Multi-rule sequences
  - `policy_events.jsonl`: Decision and outcome history
- **Markdown seeds**: `USER.md`, `SOUL.md`, `AGENTS.md` (`.clawpolicy/`)

## Contracts

`lib/contracts.py` is the single source of truth for State (17 dimensions) and Action (11 dimensions).
