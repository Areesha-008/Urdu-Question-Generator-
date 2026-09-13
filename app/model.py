import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from config import ModelConfig


class QuestionGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = c = config
        self.embedding = nn.Embedding(c.vocab_size, c.embedding, padding_idx=0)
        self.membership = nn.Embedding(3, c.feature_size)
        self.distance = nn.Embedding(2 * c.distance + 1, c.feature_size)
        self.dropout = nn.Dropout(c.dropout)
        self.encoder = nn.GRU(c.embedding + 2 * c.feature_size, c.hidden, c.layers,
            batch_first=True, bidirectional=True, dropout=c.dropout if c.layers > 1 else 0)
        self.bridge = nn.Linear(2 * c.hidden, c.hidden)
        self.answer_projection = nn.Linear(2 * c.hidden, c.embedding)
        self.decoder = nn.GRU(3 * c.embedding, c.hidden, c.layers, batch_first=True,
            dropout=c.dropout if c.layers > 1 else 0)
        self.key_projection = nn.Linear(2 * c.hidden, c.hidden, bias=False)
        self.attentional = nn.Linear(3 * c.hidden + c.embedding, c.embedding)
        self.output = nn.Linear(c.embedding, c.vocab_size)
        self.output.weight = self.embedding.weight

    def features(self, source):
        c = self.config
        opening, closing = source.eq(c.ans_open), source.eq(c.ans_close)
        if not torch.all(opening.sum(1).eq(1) & closing.sum(1).eq(1)):
            raise ValueError('Each source must have one answer span.')
        start, end = opening.long().argmax(1), closing.long().argmax(1)
        if not torch.all(end > start + 1):
            raise ValueError('The answer span is empty or reversed.')
        position = torch.arange(source.size(1), device=source.device)[None]
        inside = (position > start[:, None]) & (position < end[:, None])
        membership = inside.long() + ((position == start[:, None] + 1) & inside).long()
        distance = torch.where(position <= start[:, None], position - start[:, None] - 1,
            torch.where(position >= end[:, None], position - end[:, None] + 1, 0))
        return membership, distance.clamp(-c.distance, c.distance) + c.distance, inside

    def encode(self, source, lengths):
        membership, distance, inside = self.features(source)
        embedded = torch.cat([self.embedding(source), self.membership(membership),
            self.distance(distance)], -1)
        packed = pack_padded_sequence(self.dropout(embedded), lengths.cpu(), batch_first=True,
            enforce_sorted=False)
        outputs, hidden = self.encoder(packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True, total_length=source.size(1))
        hidden = hidden.view(self.config.layers, 2, source.size(0), -1)
        hidden = torch.tanh(self.bridge(torch.cat([hidden[:, 0], hidden[:, 1]], -1)))
        answer = (outputs * inside.unsqueeze(-1)).sum(1) / inside.sum(1, keepdim=True)
        answer = torch.tanh(self.answer_projection(answer))
        return (outputs, self.key_projection(outputs), source.ne(0), answer), hidden

    def step(self, token, hidden, feed, memory):
        outputs, keys, mask, answer = memory
        embedded = self.dropout(self.embedding(token))
        state, hidden = self.decoder(torch.cat([embedded, feed, answer], -1).unsqueeze(1), hidden)
        state = state.squeeze(1)
        scores = torch.bmm(keys, state.unsqueeze(2)).squeeze(2).masked_fill(~mask, -torch.inf)
        attention = scores.softmax(1)
        context = torch.bmm(attention.unsqueeze(1), outputs).squeeze(1)
        feed = torch.tanh(self.attentional(torch.cat([state, context, answer], -1)))
        return self.output(self.dropout(feed)), hidden, feed, attention

    def forward(self, source, lengths, target):
        memory, hidden = self.encode(source, lengths)
        feed = memory[3].new_zeros(source.size(0), self.config.embedding)
        logits = []
        for token in target[:, :-1].unbind(1):
            output, hidden, feed, _ = self.step(token, hidden, feed, memory)
            logits.append(output)
        return torch.stack(logits, 1)
