import argparse
from pathlib import Path

from config import DATA_DIR
from scripts.prepare_data import read_pairs


def validate(data_dir):
    splits = {name: read_pairs(Path(data_dir) / f'{name}.tsv')
        for name in ['train', 'valid', 'wiki_test']}
    for name, pairs in splits.items():
        if any(len(source.split()) > 60 or len(target.split()) > 25 for source, target in pairs):
            raise ValueError(f'{name}: length limit exceeded')
        if len(pairs) != len(set(pairs)):
            raise ValueError(f'{name}: duplicate pairs found')
        print(f'{name}: {len(pairs):,} pairs')
    for name in ['valid', 'wiki_test']:
        if set(splits['train']) & set(splits[name]):
            raise ValueError(f'Training pairs overlap {name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    validate(parser.parse_args().data_dir)
