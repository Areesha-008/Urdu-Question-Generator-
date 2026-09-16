import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset

from app.tokenizer import normalize, validate_source
from config import PREPARED_DIR, ROOT


def read_pairs(path):
    with open(path, encoding='utf-8', newline='') as file:
        rows = list(csv.reader(file, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\'))
    if not rows or any(len(row) != 2 or not all(row) for row in rows):
        raise ValueError(f'Invalid or empty dataset: {path}')
    for source, _ in rows:
        validate_source(source)
    return [tuple(row) for row in rows]


def write_pairs(path, pairs):
    with open(path, 'w', encoding='utf-8', newline='') as file:
        csv.writer(file, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\',
            lineterminator='\n').writerows(pairs)


def make_pair(row, max_source=60, max_target=25):
    answer, context = row.get('answer'), row['context']
    if row.get('is_impossible') or not answer:
        return None, 'unanswerable', []
    start = row['answer_start']
    if not isinstance(start, int) or start < 0 or context[start:start + len(answer)] != answer:
        return None, 'offset_mismatch', []
    boundaries = [0] + [match.end() for match in re.finditer('[۔؟!]', context)] + [len(context)]
    left, right = next((a, b) for a, b in zip(boundaries, boundaries[1:]) if a <= start < b)
    if start + len(answer) > right:
        return None, 'cross_boundary', []
    source = normalize(context[left:start] + ' <ans> ' + answer + ' </ans> '
        + context[start + len(answer):right])
    target = normalize(row['question'])
    if not target:
        return None, 'empty_target', []
    if len(source.split()) > max_source or len(target.split()) > max_target:
        return None, 'overlength', []
    flags = []
    if context.count(answer) > 1:
        flags.append('repeated_answer')
    if any(number not in source for number in re.findall(r'\d+', target)):
        flags.append('missing_number')
    return (source, target), 'kept', flags


def prepare_split(rows, output, name, decisions=None):
    output.mkdir(parents=True, exist_ok=True)
    decisions, counts, pairs, seen = decisions or {}, Counter(), [], set()
    with (output / f'{name}_audit.jsonl').open('w', encoding='utf-8') as audit:
        for index, row in enumerate(rows):
            pair, reason, flags = make_pair(row)
            key = f'{name}:{index}:{row.get("id", "")}'
            if pair and decisions.get(key) == 'drop':
                pair, reason = None, 'manual_drop'
            elif pair and pair in seen:
                pair, reason = None, 'duplicate'
            if pair:
                seen.add(pair)
                pairs.append(pair)
            counts[reason] += 1
            audit.write(json.dumps({'key': key, 'reason': reason, 'flags': flags, 'pair': pair,
                'raw': row}, ensure_ascii=False) + '\n')
    write_pairs(output / f'{name}.tsv', pairs)
    return dict(counts)


def prepare(output=PREPARED_DIR, decisions_path=ROOT / 'data' / 'curation.json'):
    decisions = json.loads(Path(decisions_path).read_text()) if Path(decisions_path).exists() else {}
    uqa = load_dataset('uqa/UQA')
    splits = [('train', uqa['train']), ('valid', uqa['validation']),
        ('wiki_test', load_dataset('uqa/Wiki-UQA')['train'])]
    statistics = {name: prepare_split(rows, Path(output), name,
        decisions if name == 'train' else {}) for name, rows in splits}
    (Path(output) / 'statistics.json').write_text(json.dumps(statistics, indent=2))
    return statistics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=PREPARED_DIR)
    parser.add_argument('--curation', type=Path, default=ROOT / 'data' / 'curation.json')
    args = parser.parse_args()
    print(prepare(args.output, args.curation))
