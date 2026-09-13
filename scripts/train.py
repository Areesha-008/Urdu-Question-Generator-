import argparse
import csv
import json
import random
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset

from app.model import QuestionGenerator
from app.tokenizer import Tokenizer, file_hash
from config import PREPARED_DIR, RUN_DIR, ModelConfig
from scripts.prepare_data import read_pairs


class QGDataset(Dataset):
    def __init__(self, path, tokenizer):
        self.pairs = read_pairs(path)
        self.items = [(torch.tensor(tokenizer.encode(source)),
            torch.tensor([2] + tokenizer.encode(target) + [3])) for source, target in self.pairs]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate(batch):
    sources, targets = zip(*batch)
    return pad_sequence(sources, batch_first=True), torch.tensor([len(source) for source in sources]), \
        pad_sequence(targets, batch_first=True)


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    total, count, norms = 0.0, 0, []
    with torch.set_grad_enabled(optimizer is not None):
        for source, lengths, target in loader:
            source, target = source.to(device), target.to(device)
            logits = model(source, lengths, target)
            tokens = target[:, 1:].ne(0).sum()
            loss = nn.functional.cross_entropy(logits.flatten(0, 1), target[:, 1:].flatten(),
                ignore_index=0, reduction='sum')
            if optimizer:
                optimizer.zero_grad(set_to_none=True)
                (loss / tokens).backward()
                norms.append(float(nn.utils.clip_grad_norm_(model.parameters(), 1.0,
                    error_if_nonfinite=True)))
                optimizer.step()
            total += loss.item()
            count += tokens.item()
    return total / count, float(np.mean(norms)) if norms else 0.0


def train(data_dir=PREPARED_DIR, run_dir=RUN_DIR, epochs=15, batch_size=64, seed=42,
          debug=False, config=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    data_dir, run_dir = Path(data_dir), Path(run_dir)
    checkpoint = run_dir / ('debug.pt' if debug else 'best.pt')
    if checkpoint.exists():
        raise FileExistsError(f'{checkpoint} exists. Choose a new run directory.')
    tokenizer = Tokenizer(run_dir / 'tokenizer.model')
    tokenizer_data = json.loads((run_dir / 'tokenizer_data.json').read_text())
    if tokenizer_data['train_sha256'] != file_hash(data_dir / 'train.tsv'):
        raise ValueError('Training data changed after tokenizer training.')
    config = config or ModelConfig(vocab_size=len(tokenizer), ans_open=tokenizer.id('<ans>'),
        ans_close=tokenizer.id('</ans>'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = QuestionGenerator(config).to(device)
    datasets = {name: QGDataset(data_dir / f'{name}.tsv', tokenizer)
        for name in ['train', 'valid']}
    training = datasets['train']
    if debug:
        training = Subset(training, random.sample(range(len(training)), min(1000, len(training))))
    train_loader = DataLoader(training, batch_size=batch_size, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(datasets['valid'], batch_size=batch_size, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, factor=0.5)
    metadata = {'schema': 2, 'versions': {name: version(name)
        for name in ['torch', 'sentencepiece', 'numpy', 'sacrebleu']}, 'config': asdict(config),
        'seed': seed, 'debug': debug, 'tokenizer_sha256': file_hash(tokenizer.path),
        'data_sha256': {name: file_hash(data_dir / f'{name}.tsv') for name in datasets},
        'parameters': sum(parameter.numel() for parameter in model.parameters()),
        'batch_size': batch_size, 'decoding': {'max_length': 60, 'beam_size': 3,
        'length_penalty': 0.6}}
    best, stale, history = float('inf'), 0, []
    for epoch in range(1, epochs + 1):
        train_loss, norm = run_epoch(model, train_loader, device, optimizer)
        valid_loss, _ = run_epoch(model, valid_loader, device)
        row = {'epoch': epoch, 'train_loss': train_loss, 'valid_loss': valid_loss,
            'gradient_norm': norm, 'lr': optimizer.param_groups[0]['lr']}
        history.append(row)
        print(row, flush=True)
        if valid_loss < best:
            best, stale = valid_loss, 0
            torch.save({'model': model.state_dict(), 'metadata': metadata, 'epoch': epoch,
                'valid_loss': best}, checkpoint)
        else:
            stale += 1
        history_path = run_dir / ('debug_history.csv' if debug else 'history.csv')
        with history_path.open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=row)
            writer.writeheader()
            writer.writerows(history)
        scheduler.step(valid_loss)
        if stale >= 4:
            break
    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=PREPARED_DIR)
    parser.add_argument('--run-dir', type=Path, default=RUN_DIR)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    train(args.data_dir, args.run_dir, args.epochs, args.batch_size, debug=args.debug)
