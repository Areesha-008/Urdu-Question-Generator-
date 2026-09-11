# Contributing

## Local setup

Use Python 3.9 or newer and install the project dependencies in an isolated
environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Before committing

Run the data validator whenever a TSV split or its generation logic changes:

```bash
python scripts/validate_data.py
```

Keep notebook outputs cleared unless an output is the explicit subject of the
change. Do not commit local environments, generated corpora, checkpoints,
experiment logs, or caches; the repository's `.gitignore` lists these paths.

## Project conventions

- `data/` contains version-controlled dataset splits.
- `notebooks/` contains exploratory work and phase prototypes.
- `scripts/` contains repeatable maintenance checks.
- `docs/` contains project decisions and reports.
- Store the frozen SentencePiece model and vocabulary in
  `artifacts/tokenizer/`; commit those two files once selected, but never the
  regenerated tokenizer corpus.
