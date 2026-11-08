## What this changes

<!-- One or two sentences. Link the issue if there is one. -->

## Checklist

- [ ] No real transaction data, account numbers or names anywhere in the change
- [ ] Each new or changed detector matches a brute-force enumeration of its definition in `tests/test_bruteforce.py`
- [ ] Deterministic: no randomness, no model, results independent of input row order
- [ ] `uv run pytest`, `ruff check`, `ruff format --check` and `mypy src` pass locally
