# Urdu Question Generator

A sequence-to-sequence model that generates an Urdu question given a context
sentence and a highlighted answer span.

**Task formulation**

| | |
| :-- | :-- |
| **Source** | The sentence containing the answer, with the span wrapped in `<ans> … </ans>` |
| **Target** | The Urdu question whose answer is that span |

```text
Source:  ہیوسٹن ، ٹیکساس میں پیدا ہوئی … اور <ans> 1990 کی دہائی کے آخر میں </ans> R&B گرل گروپ … شہرت حاصل کی۔
Target:  بیونس نے کب مقبولیت حاصل کرنا شروع کی؟
```

## Status

| Phase | Scope | State |
| :-- | :-- | :-- |
| 1 | Data preparation & exploration | Done — [`notebooks/data_prep.ipynb`](notebooks/data_prep.ipynb), [`docs/phase1_data_prep.md`](docs/phase1_data_prep.md) |
| 2 | SentencePiece tokenizer | Notebook written — [`notebooks/tokenizer.ipynb`](notebooks/tokenizer.ipynb); artifacts not yet committed |
| 3 | Seq2seq training | Data-loading scaffold written — [`notebooks/training.ipynb`](notebooks/training.ipynb); training loop not yet implemented |
| 4 | Evaluation | Not started |

See [`docs/repo_audit.md`](docs/repo_audit.md) for the current audit of the
pipeline and the open issues to resolve before Phase 3.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.9 or newer.

## Data

The three splits are committed under `data/` so the pipeline does not have to be
re-run to train a model:

| File | Pairs | Purpose |
| :-- | --: | :-- |
| `data/train.tsv` | 75,067 | Training |
| `data/valid.tsv` | 10,018 | Checkpoint selection |
| `data/wiki_test.tsv` | 177 | Out-of-domain test (Urdu Wikipedia) |

Format: two tab-separated columns, no header, `source<TAB>target`, written with
`csv.QUOTE_NONE` and `escapechar="\\"`. **Read them with `csv.reader` using those
same settings** — a naive `line.split("\t")` mis-parses the rows that contain a
literal backslash.

```python
import csv

with open("data/train.tsv", encoding="utf-8", newline="") as f:
    pairs = list(csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\"))
```

To regenerate them from scratch, run [`notebooks/data_prep.ipynb`](notebooks/data_prep.ipynb),
which pulls [`uqa/UQA`](https://huggingface.co/datasets/uqa/UQA) and
[`uqa/Wiki-UQA`](https://huggingface.co/datasets/uqa/Wiki-UQA) from the Hugging
Face Hub.

## Validating the data

Any change that regenerates the TSVs should be checked before it is committed:

```bash
python scripts/validate_data.py
```

It verifies row structure, `<ans>` tag integrity, length limits, and cross-split
leakage, and exits non-zero on a hard failure.

## Repository layout

```
data/            Committed TSV splits (source<TAB>target)
docs/            Project notes, phase write-ups, and the repository audit
notebooks/       Exploratory and phase notebooks
scripts/         Maintenance and validation scripts
requirements.txt Pinned-by-lower-bound dependencies
```

## Conventions

- **Tokenizer artifacts are committed, `corpus.txt` is not.** Retraining
  SentencePiece reshuffles token IDs and silently invalidates any checkpoint
  trained against the old vocabulary, so `artifacts/tokenizer/ur_sp.model` and
  `.vocab` are
  version-controlled; the 25 MB corpus they are trained from is regenerated.
- **Clear notebook outputs before committing** unless a plot is the point of the
  commit — output blobs make notebook diffs unreviewable.
- **Do not evaluate Urdu with `rouge-score`'s default tokenizer.** It strips every
  non-`[a-z0-9]` character and returns 0.0 even when the hypothesis matches the
  reference exactly. Use `sacrebleu` (BLEU and chrF), or pass a custom Urdu
  tokenizer to `rouge_scorer`.
