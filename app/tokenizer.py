import hashlib
import re
import unicodedata
from pathlib import Path

import sentencepiece as spm


def normalize(text):
    return ' '.join(unicodedata.normalize('NFC', text).split())


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mark_answer(sentence, answer):
    sentence, answer = normalize(sentence), normalize(answer)
    if not answer or '<ans>' in sentence or '</ans>' in sentence:
        raise ValueError('Enter an unmarked sentence and a nonempty answer.')
    match = re.search(re.escape(answer), sentence)
    if not match:
        raise ValueError('The answer does not appear in the sentence.')
    return normalize(sentence[:match.start()] + ' <ans> ' + answer + ' </ans> ' + sentence[match.end():])


def validate_source(source):
    source = normalize(source)
    if source.count('<ans>') != 1 or source.count('</ans>') != 1:
        raise ValueError('Mark exactly one answer with <ans> and </ans>.')
    start, end = source.index('<ans>') + 5, source.index('</ans>')
    if start >= end or not source[start:end].strip():
        raise ValueError('The marked answer must be nonempty and ordered.')
    return source


class Tokenizer:
    def __init__(self, path):
        self.path = Path(path)
        self.processor = spm.SentencePieceProcessor(model_file=str(path))

    def __len__(self):
        return len(self.processor)

    def encode(self, text, pieces=False):
        return self.processor.encode(normalize(text), out_type=str if pieces else int)

    def decode(self, ids):
        return self.processor.decode(ids)

    def id(self, piece):
        return self.processor.piece_to_id(piece)
