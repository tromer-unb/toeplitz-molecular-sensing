# Contributing

Changes should preserve scientific traceability. Use focused branches such as `figure/2-identifiability`, `model/decoder`, or `docs/reproducibility` and open a pull request into `main`.

Before opening a pull request:

```bash
python -m pip install -e '.[dev]'
ruff check src tests
pytest
```

For any change that modifies a reported result, include the manuscript section/figure affected, the random seed or seed rule, the command used, and a short before/after numerical summary. Do not commit virtual environments, caches, or ad-hoc local outputs.
