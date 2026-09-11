# Notebooks

Notebooks are organized by project phase:

1. `data_prep.ipynb` downloads, filters, and exports the TSV splits.
2. `tokenizer.ipynb` creates the SentencePiece tokenizer.
3. `training.ipynb` currently verifies dataset loading and batching.

Run them from the repository root where possible so relative paths resolve
consistently. Clear outputs before committing. Generated tokenizer corpora,
training checkpoints, and experiment logs are intentionally ignored; the final
tokenizer model and vocabulary belong in `artifacts/tokenizer/` once frozen.
