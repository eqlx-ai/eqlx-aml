# Contributing to eqlx-aml

Thanks for helping. Bug reports, new patterns, faster algorithms and documentation fixes are welcome.

## Ground rules

1. **Never share real transaction data.** No real accounts, names, IBANs or amounts in issues, tests or
   examples. Describe the structure, or reproduce it with `eqlx_aml.synthetic`.
2. **Deterministic only.** A detector is an exact, written definition with explicit parameters: no
   model, no randomness, no learned threshold. Same input, same output, in the same order.
3. **Every detector has a brute-force twin.** `tests/test_bruteforce.py` checks each detector against a
   direct enumeration of its definition on many small random graphs. A new or changed detector must
   pass the same kind of test, plus a synthetic case that triggers it and one that does not.
4. **Public sources.** Cite the typology (FATF, FinCEN, a regulator's guidance, a paper) a pattern
   follows. If it can be measured on the IBM AML data, add it to `benchmarks/ibm_aml.py`.
5. **Dependencies:** NumPy for the core, `mcp` for the server. Propose anything else in an issue first.

## Workflow

1. Open an issue first for anything larger than a small fix.
2. Fork, create a branch (`fix/...`, `feat/...`), and run the checks locally:

   ```bash
   uv sync --all-extras
   uv run pytest
   uv run ruff check . && uv run ruff format --check . && uv run mypy src
   ```

3. Open a pull request against `main` and fill in the template. CI must pass before review; for
   outside contributors a maintainer approves the CI run.

By contributing you agree that your contribution is licensed under the Apache License 2.0.

Questions: open an issue or write to research@eqlx.ai.
