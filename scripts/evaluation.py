import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import sacrebleu
import torch
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from app.inference import Generator
from app.tokenizer import file_hash
from config import PREPARED_DIR, RUN_DIR
from scripts.train import QGDataset, collate, run_epoch


class UrduTokenizer:
    def tokenize(self, text):
        return text.split()


def evaluate(data_dir=PREPARED_DIR, run_dir=RUN_DIR, output=None):
    data_dir, run_dir = Path(data_dir), Path(run_dir)
    output = Path(output or run_dir / 'results')
    output.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator = Generator(run_dir, device)
    rouge = rouge_scorer.RougeScorer(['rougeL'], tokenizer=UrduTokenizer())
    metrics, samples = [], []

    for split in ['valid', 'wiki_test']:
        dataset = QGDataset(data_dir / f'{split}.tsv', generator.tokenizer)
        if split == 'valid' and file_hash(data_dir / 'valid.tsv') != \
                generator.metadata['data_sha256']['valid']:
            raise ValueError('Validation data differs from the training run.')
        loss, _ = run_epoch(generator.model,
            DataLoader(dataset, batch_size=32, collate_fn=collate), device)
        references = defaultdict(list)
        for source, target in dataset.pairs:
            if target not in references[source]:
                references[source].append(target)
        predictions = {source: {name: generator.generate(source, beam_size=size)
            for name, size in [('greedy', 1), ('beam', 3)]}
            for source in tqdm(references, desc=split)}
        with (output / f'{split}_predictions.jsonl').open('w', encoding='utf-8') as file:
            for source, refs in references.items():
                file.write(json.dumps({'source': source, 'references': refs,
                    **predictions[source]}, ensure_ascii=False) + '\n')

        for decoding in ['greedy', 'beam']:
            for protocol in ['single_reference', 'grouped_multi_reference']:
                sources = [source for source, _ in dataset.pairs] if protocol == \
                    'single_reference' else list(references)
                refs = [[target] for _, target in dataset.pairs] if protocol == \
                    'single_reference' else list(references.values())
                streams = [[row[i] if i < len(row) else None for row in refs]
                    for i in range(max(map(len, refs)))]
                hypotheses = [predictions[source][decoding]['text'] for source in sources]
                generated = [predictions[source][decoding] for source in sources]
                bleu = sacrebleu.metrics.BLEU()
                score = bleu.corpus_score(hypotheses, streams)
                metrics.append({'split': split, 'decoding': decoding, 'protocol': protocol,
                    'bleu4': score.score, 'chrf': sacrebleu.corpus_chrf(hypotheses, streams).score,
                    'rougeL': sum(max(rouge.score(ref, hypothesis)['rougeL'].fmeasure
                    for ref in row) for hypothesis, row in zip(hypotheses, refs)) / len(hypotheses),
                    'perplexity': math.exp(loss),
                    'unk_rate': sum(row['ids'].count(1) for row in generated) /
                    max(1, sum(len(row['ids']) for row in generated)),
                    'eos_rate': sum(row['ended'] for row in generated) / len(generated),
                    'length_ratio': score.sys_len / max(1, score.ref_len),
                    'signature': str(bleu.get_signature())})

        if split == 'valid':
            chosen = random.Random(42).sample(list(references), min(50, len(references)))
            samples = [{'id': index, 'source': source, 'reference': references[source][0],
                'greedy': predictions[source]['greedy']['text'],
                'beam': predictions[source]['beam']['text']}
                for index, source in enumerate(chosen)]

    trace = generator.generate(samples[0]['source'], trace=True)
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.imshow(trace['attention'], aspect='auto')
    axis.set(xlabel='Source subword index', ylabel='Generated subword index')
    figure.tight_layout()
    figure.savefig(output / 'attention.png', dpi=150)
    plt.close(figure)
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    with (output / 'metrics.tsv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=metrics[0], delimiter='\t')
        writer.writeheader()
        writer.writerows(metrics)
    with (output / 'samples.tsv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=samples[0], delimiter='\t')
        writer.writeheader()
        writer.writerows(samples)
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=PREPARED_DIR)
    parser.add_argument('--run-dir', type=Path, default=RUN_DIR)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    for row in evaluate(args.data_dir, args.run_dir, args.output):
        print(row)
