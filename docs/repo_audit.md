# Repository Audit — 2026-09-09

Audit of everything committed through `ce46434` ("uploaded tokenizer code"):
the Phase 1 data pipeline, the committed TSV splits, the Phase 2 tokenizer
notebook, and repository hygiene.

Findings are ordered by severity. Each one says what is wrong, why it matters,
and what to do. Items marked **[fixed]** were resolved in this audit pass; the
rest are open and need a decision from the team.

---

## High severity

### E1. `rouge-score` silently reports 0.0 on Urdu — **open**

`docs/phase1_data_prep.md` lists `rouge-score` as an evaluation metric. Its
default tokenizer lowercases and then applies `re.sub(r"[^a-z0-9]+", " ", text)`,
which deletes every Urdu character. Verified empirically:

```python
>>> from rouge_score import rouge_scorer, tokenize
>>> tokenize.tokenize("بیونس نے کب مقبولیت حاصل کرنا شروع کی؟", None)
[]
>>> scorer.score(ref, ref)          # identical strings
{'rouge1': Score(precision=0.0, recall=0.0, fmeasure=0.0), ...}
```

ROUGE returns 0.0 even when the hypothesis is character-for-character identical
to the reference. If Phase 4 reports ROUGE without noticing this, every number in
the results table is meaningless.

**Fix:** drop ROUGE, or pass a custom Urdu tokenizer to `RougeScorer(...,
tokenizer=...)`. `sacrebleu` is unaffected — BLEU (`13a` and `intl`) and chrF both
produce sensible scores on the same pair (24.6 / 24.2 / 41.5). For a
morphologically rich language, report **chrF alongside BLEU**.

### D1. `valid.tsv` is 17.7% exact duplicate rows — **open**

1,769 of 10,018 validation rows are byte-identical `(source, target)` pairs;
only 8,249 are unique. The cause is upstream: SQuAD 2.0's dev set stores several
gold answer annotations per question, and after span tagging many collapse onto
the same sentence and the same question.

**Why it matters:** validation loss and BLEU end up weighted by how many
annotators happened to answer each question. A question annotated five times
counts five times toward checkpoint selection. Some rows repeat 5×.

**Fix:** deduplicate `valid.tsv` to its 8,249 unique pairs. No information is
lost — the duplicates are identical strings. `train.tsv` has 67 duplicates
(0.1%), harmless but worth removing in the same pass. `wiki_test.tsv` is clean.

### T2. Tokenizer artifacts are not committed — **open**

`notebooks/tokenizer.ipynb` trains `artifacts/tokenizer/ur_sp.model` / `.vocab`, but neither is
in the repository. SentencePiece training is not deterministic across corpus or
version changes, so two people running the notebook get **different token IDs for
the same text**. Any checkpoint trained against one vocabulary is silently
incompatible with the other.

**Fix:** run the notebook once and commit `artifacts/tokenizer/ur_sp.model` and
`artifacts/tokenizer/ur_sp.vocab` (~400 KB). Treat them as frozen for the rest of the project.
`.gitignore` already excludes the 25 MB `corpus.txt` they are built from.

---

## Medium severity

### D3. ~7,951 examples are dropped with no explanation — **open**

Phase 1 reports 83,018 answerable training rows and emits 75,067 pairs. The
missing 9.6% are discarded by `make_pair`, which returns `None` for three
different reasons — offset-integrity mismatch, an answer that straddles a
sentence boundary, and over-length source or target — without recording which.

**Why it matters:** if most of the loss is offset mismatch, that points at a
translation-alignment problem in the source dataset worth quantifying in the
write-up. If it is mostly length filtering, raising `max_src` recovers real data.
Right now nobody can tell.

**Fix:** have `make_pair` return a rejection reason and print a breakdown.

### D2. `is_impossible` is documented as a filter but never read — **open**

`docs/phase1_data_prep.md` §4 Step 1 states that rows with `is_impossible ==
True` are filtered out. `make_pair` only checks whether the `answer` string is
non-empty; `is_impossible` appears nowhere in the code. The two probably coincide
in this dataset, but the claim in the write-up is currently unverified.

**Fix:** either add the explicit check, or correct the write-up to describe what
the code actually does.

### T4. The fertility check promised by the notebook does not exist — **open**

The `tokenizer.ipynb` header advertises "subword fertility" as step 4 of
verification. The final cell checks vocabulary size, `<ans>` atomicity, and
decode round-tripping — all useful — but never computes fertility.

**Why it matters:** fertility (subwords per whitespace word) is the number that
justifies `vocab_size=8000`. Without it there is no evidence the vocabulary is
the right size for Urdu, and an over-fragmented vocabulary inflates sequence
lengths and training cost.

**Fix:** add a cell measuring mean pieces-per-word over a validation sample, and
sanity-check 8k against 4k/16k before Phase 3 locks the vocabulary in.

### E2. Single-reference BLEU understates quality — **open**

2,270 training sources and 326 validation sources are paired with more than one
distinct gold question. This is legitimate — several valid questions can target
the same span — but scoring each row independently penalises a model that
produces the *other* valid question.

**Fix:** in the evaluation script, group by source and pass all gold questions as
multiple references. `sacrebleu` supports this directly.

---

## Low severity

### D4. `<ans>` tags count toward the 60-word source limit

The tags are inserted before the length filter runs, so the real sentence budget
is 58 words, not 60. Harmless, but the write-up should say so.

### D5. 192 training targets are ≤ 3 words

Mostly formulaic definitional questions (`سامسارا کیا ہے؟`). Legitimate Urdu
questions, not corruption — flagged only so the distribution is not a surprise
later.

### L1. Three sources are shared between train and valid

0 identical `(source, target)` pairs overlap, and only 3 source sentences out of
75,067 appear in both splits. Negligible; recorded so it is not re-discovered as
a bug.

### H4. Notebook outputs are committed

`notebooks/data_prep.ipynb` carries 126 KB of embedded output, mostly base64 PNG
histograms. Every re-run produces a large unreadable diff.

**Fix:** clear outputs before committing, or add `nbstripout` as a pre-commit
hook. Keeping the histograms is a reasonable exception if they are the point of
the commit — the problem is the default, not the plots.

### H5. Empty `app.py` placeholder — **resolved**

The zero-byte inference placeholder was not referenced by the project and has
been removed. Add an application entry point only when an inference interface is
implemented.

### H8. Notebooks `pip install` inline instead of using `requirements.txt`

`data_prep.ipynb` cell 0 and `tokenizer.ipynb` cell 1 install unpinned packages.
Convenient in Colab, but it means the environment is defined in three places.
Point them at `requirements.txt` (`%pip install -r ../requirements.txt`).

---

## Fixed in this pass

| ID | Issue | Resolution |
| :-- | :-- | :-- |
| H1 | No `.gitignore` — the next person to run `tokenizer.ipynb` would have committed a 25 MB `corpus.txt` | Added, covering generated corpora, checkpoints, caches, and macOS/editor cruft |
| H2 | `requirements.txt` empty while three notebooks depend on eight packages | Populated with lower-bound constraints and a note about the ROUGE trap |
| H3 | `README.md` was a two-line stub | Rewritten: task definition, phase status, setup, data format, layout, conventions |
| H6 | No `.gitattributes` — a Windows checkout could CRLF-convert the TSVs and shift character offsets | Added, forcing LF on all text and marking `*.model` binary |
| H7 | No way to check the data after regenerating it | Added `scripts/validate_data.py` — structure, tag integrity, length limits, duplicates, and cross-split leakage; non-zero exit on hard failures |

---

## What good looks like from here

1. Deduplicate `valid.tsv` (**D1**), then re-run `scripts/validate_data.py`.
2. Run `tokenizer.ipynb` once and commit `ur_sp.model` / `.vocab` (**T2**). Freeze them.
3. Add the fertility measurement and confirm 8k is the right vocabulary size (**T4**)
   *before* any checkpoint depends on it.
4. Settle the metric set (**E1**, **E2**) before writing the training loop, so
   Phase 3 reports numbers that mean something.
5. Instrument `make_pair` with rejection reasons (**D3**) and update the Phase 1
   write-up with the breakdown.
