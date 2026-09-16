import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sentencepiece as spm

from app.tokenizer import Tokenizer, file_hash
from config import PREPARED_DIR, RUN_DIR
from scripts.prepare_data import read_pairs


def train_tokenizer(train_path, run_dir, vocab_size=8000):
    train_path, run_dir = Path(train_path), Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / 'tokenizer.model'
    if model_path.exists():
        raise FileExistsError('Use a new run directory instead of replacing the tokenizer.')
    corpus = run_dir / 'corpus.txt'
    corpus.write_text('\n'.join(text for pair in read_pairs(train_path) for text in pair),
        encoding='utf-8')
    spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(run_dir / 'tokenizer'),
        vocab_size=vocab_size, model_type='unigram', character_coverage=1.0,
        user_defined_symbols=['<ans>', '</ans>'], pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        shuffle_input_sentence=False, num_threads=1, hard_vocab_limit=False)
    (run_dir / 'tokenizer_data.json').write_text(json.dumps({'train_sha256': file_hash(train_path)}))
    corpus.unlink()
    return Tokenizer(model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path, default=PREPARED_DIR / 'train.tsv')
    parser.add_argument('--run-dir', type=Path, default=RUN_DIR)
    parser.add_argument('--vocab-size', type=int, default=8000)
    args = parser.parse_args()
    train_tokenizer(args.train, args.run_dir, args.vocab_size)
