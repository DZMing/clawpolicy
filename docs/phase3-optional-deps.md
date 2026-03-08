# Phase3 Optional Dependencies

Install with:

```bash
pip install "clawpolicy[phase3]"
```

Feature mapping:

- `redis`, `celery`: distributed training
- `torch`, `tensorboard`: neural models and monitoring
- `numba`: JIT optimization
- `scipy`: advanced curve fitting in metrics analysis

The codebase degrades gracefully when these dependencies are missing.
