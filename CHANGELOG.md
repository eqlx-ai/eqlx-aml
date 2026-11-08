# Changelog

## 0.1.0

- Read transfers from CSV (generic columns, found by name or given, and the IBM AML layout) or from
  Python records. Amounts are exact integers in millionths; no floating point.
- Detectors: fan-out, fan-in, gather-scatter, scatter-gather, cycles (time-respecting loops),
  pass-through and structuring. Each is checked against a brute-force enumeration of its definition
  on random graphs. Results do not depend on the order of input rows.
- Account profile and time-respecting fund tracing (forward and backward).
- Command line `aml` and a read-only MCP server (seven tools, one folder).
- Benchmark on the public IBM AML data (`benchmarks/ibm_aml.py`), synthetic data generator.
