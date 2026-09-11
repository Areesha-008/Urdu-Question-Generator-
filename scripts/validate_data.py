#!/usr/bin/env python3
"""Validate the Urdu QG datasets in data/.

Run before committing any change that regenerates the TSVs:

    python scripts/validate_data.py

Exits non-zero if a hard invariant is broken (malformed rows, missing <ans>
tags, empty fields, cross-split leakage). Soft issues — duplicates, length
outliers — are reported as warnings so they stay visible without blocking.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

ANS_OPEN = "<ans>"
ANS_CLOSE = "</ans>"

# Must match the filters in notebooks/data_prep.ipynb.
MAX_SRC_WORDS = 60
MAX_TGT_WORDS = 25

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
SPLITS = ("train", "valid", "wiki_test")

errors: list[str] = []
warnings: list[str] = []


def fail(msg: str) -> None:
    errors.append(msg)
    print(f"  FAIL  {msg}")


def warn(msg: str) -> None:
    warnings.append(msg)
    print(f"  WARN  {msg}")


def ok(msg: str) -> None:
    print(f"  ok    {msg}")


def read_split(path: Path) -> list[tuple[str, str]]:
    """Read a TSV written with QUOTE_NONE + escapechar='\\'.

    Rows that do not have exactly two columns are a hard error rather than a
    silent skip — silently dropping them is how a corrupt split goes unnoticed.
    """
    pairs: list[tuple[str, str]] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        for lineno, row in enumerate(reader, start=1):
            if len(row) != 2:
                fail(f"{path.name}:{lineno} has {len(row)} columns, expected 2")
                continue
            pairs.append((row[0], row[1]))
    return pairs


def check_split(name: str, pairs: list[tuple[str, str]]) -> None:
    print(f"\n{name}.tsv — {len(pairs):,} pairs")

    bad_tags = sum(
        s.count(ANS_OPEN) != 1 or s.count(ANS_CLOSE) != 1 for s, _ in pairs
    )
    if bad_tags:
        fail(f"{bad_tags} source(s) do not contain exactly one {ANS_OPEN}…{ANS_CLOSE} pair")
    else:
        ok("every source has exactly one <ans>…</ans> pair")

    misordered = sum(s.find(ANS_OPEN) > s.find(ANS_CLOSE) for s, _ in pairs)
    if misordered:
        fail(f"{misordered} source(s) have </ans> before <ans>")

    empty = sum(not s.strip() or not t.strip() for s, t in pairs)
    if empty:
        fail(f"{empty} row(s) have an empty source or target")
    else:
        ok("no empty sources or targets")

    over_src = sum(len(s.split()) > MAX_SRC_WORDS for s, _ in pairs)
    over_tgt = sum(len(t.split()) > MAX_TGT_WORDS for _, t in pairs)
    if over_src or over_tgt:
        fail(
            f"{over_src} source(s) > {MAX_SRC_WORDS} words, "
            f"{over_tgt} target(s) > {MAX_TGT_WORDS} words"
        )
    else:
        ok(f"all rows within {MAX_SRC_WORDS}/{MAX_TGT_WORDS} word limits")

    dupes = len(pairs) - len(set(pairs))
    if dupes:
        pct = 100 * dupes / len(pairs)
        warn(f"{dupes:,} exact duplicate (source, target) rows ({pct:.1f}% of split)")
    else:
        ok("no exact duplicate rows")

    # A source paired with several different questions is legitimate for QG,
    # but it means single-reference BLEU understates quality — flag it so the
    # evaluation script can group references instead.
    by_src: dict[str, set[str]] = defaultdict(set)
    for s, t in pairs:
        by_src[s].add(t)
    multi = sum(1 for refs in by_src.values() if len(refs) > 1)
    if multi:
        print(
            f"  note  {multi:,} source(s) have >1 distinct question "
            f"— use multi-reference BLEU on this split"
        )

    short_tgt = sum(len(t.split()) <= 3 for _, t in pairs)
    if short_tgt:
        print(f"  note  {short_tgt:,} target(s) are <= 3 words")


def check_leakage(splits: dict[str, list[tuple[str, str]]]) -> None:
    print("\ncross-split leakage")
    names = [n for n in SPLITS if n in splits]
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared_pairs = set(splits[a]) & set(splits[b])
            if shared_pairs:
                fail(f"{a} and {b} share {len(shared_pairs):,} identical (source, target) pairs")
            shared_src = {s for s, _ in splits[a]} & {s for s, _ in splits[b]}
            if shared_src:
                warn(f"{a} and {b} share {len(shared_src):,} identical source sentences")
            if not shared_pairs and not shared_src:
                ok(f"{a} vs {b}: no shared sources or pairs")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=DATA_DIR, help="directory holding the TSV splits"
    )
    args = parser.parse_args()

    splits: dict[str, list[tuple[str, str]]] = {}
    for name in SPLITS:
        path = args.data_dir / f"{name}.tsv"
        if not path.exists():
            warn(f"{path} not found — skipping")
            continue
        splits[name] = read_split(path)
        check_split(name, splits[name])

    if not splits:
        print("\nNo splits found. Run notebooks/data_prep.ipynb first.")
        return 1

    if len(splits) > 1:
        check_leakage(splits)

    print(f"\n{'-' * 60}")
    print(f"{len(errors)} error(s), {len(warnings)} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
