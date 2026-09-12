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
| 1 | Data preparation & exploration | Done — [`notebooks/data_prep.ipynb`](notebooks/data_prep.ipynb) |
| 2 | SentencePiece tokenizer | Implemented — [`notebooks/tokenizer.ipynb`](notebooks/tokenizer.ipynb) |
| 3 | GRU/Luong model | Implemented — [`notebooks/model.ipynb`](notebooks/model.ipynb); retraining required after the latest improvements |
| 4 | Evaluation | Implemented — [`notebooks/evaluation.ipynb`](notebooks/evaluation.ipynb) |

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
| `data/train.tsv` | 75,000 | Training |
| `data/valid.tsv` | 8,249 | Checkpoint selection |
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
notebooks/       Exploratory and phase notebooks
scripts/         Maintenance and validation scripts
requirements.txt Pinned-by-lower-bound dependencies
```

## Conventions

- **Keep each tokenizer with its checkpoint.** Retraining SentencePiece reshuffles
  token IDs and silently invalidates old checkpoints. The tokenizer and model
  artifacts are saved together on Google Drive; the generated corpus is ignored.
- **Clear notebook outputs before committing** unless a plot is the point of the
  commit — output blobs make notebook diffs unreviewable.
- **Do not evaluate Urdu with `rouge-score`'s default tokenizer.** It strips every
  non-`[a-z0-9]` character and returns 0.0 even when the hypothesis matches the
  reference exactly. Use `sacrebleu` (BLEU and chrF), or pass a custom Urdu
  tokenizer to `rouge_scorer`.
