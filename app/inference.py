from pathlib import Path

import torch

from app.model import QuestionGenerator
from app.tokenizer import Tokenizer, file_hash, validate_source
from config import ModelConfig


class Generator:
    def __init__(self, run_dir, device='cpu'):
        run_dir = Path(run_dir)
        checkpoint = torch.load(run_dir / 'best.pt', map_location='cpu', weights_only=True)
        self.metadata = checkpoint['metadata']
        tokenizer_path = run_dir / 'tokenizer.model'
        if self.metadata['debug'] or file_hash(tokenizer_path) != self.metadata['tokenizer_sha256']:
            raise ValueError('Checkpoint and tokenizer do not match.')
        self.tokenizer = Tokenizer(tokenizer_path)
        self.device = torch.device(device)
        self.model = QuestionGenerator(ModelConfig(**self.metadata['config'])).to(self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()

    @torch.inference_mode()
    def generate(self, source, beam_size=1, max_length=60, length_penalty=0.6, trace=False):
        ids = self.tokenizer.encode(validate_source(source))
        if len(ids) > 256:
            raise ValueError('Use a sentence with at most 256 subwords.')
        source = torch.tensor([ids], device=self.device)
        memory, hidden = self.model.encode(source, torch.tensor([len(ids)]))
        feed = memory[3].new_zeros(1, self.model.config.embedding)
        traces, beams = [], [([], 0.0, hidden, feed, False)]

        def rank(beam):
            length = ((5 + max(1, len(beam[0]))) / 6) ** length_penalty
            return beam[1] / length

        for _ in range(max_length):
            candidates = []
            for tokens, score, state, feed, ended in beams:
                if ended:
                    candidates.append((tokens, score, state, feed, True))
                    continue
                token = torch.tensor([tokens[-1] if tokens else 2], device=self.device)
                logits, state, next_feed, attention = self.model.step(token, state, feed, memory)
                if trace and beam_size == 1:
                    traces.append(attention[0].cpu().tolist())
                logits[:, [0, 2, self.model.config.ans_open, self.model.config.ans_close]] = -torch.inf
                values, indices = logits.log_softmax(-1).topk(beam_size)
                for value, index in zip(values[0].tolist(), indices[0].tolist()):
                    candidates.append((tokens + [index], score + value, state, next_feed, index == 3))
            beams = sorted(candidates, key=rank, reverse=True)[:beam_size]
            if all(beam[4] for beam in beams):
                break
        best = max([beam for beam in beams if beam[4]] or beams, key=rank)
        tokens = best[0][:-1] if best[4] else best[0]
        result = {'text': self.tokenizer.decode(tokens), 'ids': tokens, 'ended': best[4]}
        if trace and beam_size == 1:
            result['attention'] = traces[:len(tokens)]
        return result
