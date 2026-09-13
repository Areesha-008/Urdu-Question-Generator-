from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'
PREPARED_DIR = ROOT / 'artifacts' / 'data'
RUN_DIR = ROOT / 'artifacts' / 'answer_gru_v1'


@dataclass
class ModelConfig:
    vocab_size: int = 8000
    embedding: int = 256
    hidden: int = 512
    layers: int = 2
    dropout: float = 0.3
    feature_size: int = 16
    distance: int = 16
    ans_open: int = 4
    ans_close: int = 5
