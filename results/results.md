# Results

## Table 1 - Dataset statistics

|  | Train | Validation | Wiki-UQA |
| --- | ---: | ---: | ---: |
| Rows in raw dataset | 124,745 | 16,824 | 210 |
| Answerable rows | 83,018 | 11,169 | 210 |
| Pairs after length filter | 74,999 | 8,249 | 177 |
| Mean source / target length | 32.59 / 11.92 | 33.25 / 12.31 | 31.69 / 11.42 |

Lengths are whitespace-token counts. Sources are limited to 60 tokens and targets to 25 tokens.

## Table 2 - Model configuration

| Field | Value |
| --- | --- |
| Encoder / decoder type | 2-layer bidirectional GRU / 2-layer GRU with Luong attention |
| Layers / embedding / hidden size | 2 / 256 / 512 |
| Vocabulary size | 8,000 SentencePiece subwords |
| Trainable parameters | 14,560,640 |
| Optimiser, learning rate, schedule | AdamW, 0.001, ReduceLROnPlateau (factor 0.5, patience 1) |
| Batch size, epochs, wall-clock, GPU | 64, 15, 53:00, NVIDIA T4 |
| Decoding | Greedy and beam search, beam size 3 |

## Table 3 - Automatic metrics

| Split | Decoding | BLEU-4 | ROUGE-L | PPL | <unk> % |
| --- | --- | ---: | ---: | ---: | ---: |
| UQA valid | greedy | 10.16 | 0.290 | 19.61 | 0.00 |
| UQA valid | beam (k=3) | 11.29 | 0.311 | 19.61 | 0.00 |
| Wiki-UQA | greedy | 8.09 | 0.261 | 22.03 | 0.10 |
| Wiki-UQA | beam (k=3) | 9.23 | 0.295 | 22.03 | 0.08 |

Grouped multi-reference BLEU-4 is 10.43 (greedy) and 11.65 (beam) on validation, and 8.18 (greedy) and 9.37 (beam) on Wiki-UQA.

## Table 4 - Human evaluation (50 samples)

|  | Fluency | Relevance | Answerability |
| --- | ---: | ---: | ---: |
| Member 1 (% yes) | - | - | - |
| Member 2 (% yes) | - | - | - |
| Cohen's kappa | - | - | - |

Human ratings are pending.

## Figures

- [Model and application architecture](figures/architecture_diagram.png)
- [Training and validation loss](figures/loss_curve.png)
- [Source and target length histograms](figures/length_histograms.png)
- [Attention heatmap](figures/attention_heatmap.png)
- [Frontend screenshot](figures/frontend.png)

The plotted loss values are preserved in [history.csv](history.csv).

## Qualitative samples

The 50 fixed-seed validation examples are in [samples.tsv](samples.tsv). Selected good and bad examples, with failure types for the bad outputs, are in [qualitative_labels.tsv](qualitative_labels.tsv).

## Tokenizer examples

Five tokenized source-target examples are in [tokenizer_examples.tsv](tokenizer_examples.tsv). SentencePiece splits Urdu words into reusable subword pieces while preserving the answer boundary markers as dedicated symbols.

## Discussion

The model handles direct what, who, when, and how much question patterns most reliably when the marked answer is explicit and close to the relevant predicate. Beam search improves answer-focused wording on several examples, but can also preserve a fluent-looking question with the wrong entity or question type. Wiki-UQA scores are lower than UQA validation because it is out of domain and contains translated text with different wording and topic distributions from the training data. The gap is consistent with reduced lexical and structural coverage rather than train-validation leakage: validation BLEU-4 is 10.16 greedy and 11.29 beam, while Wiki-UQA is 8.09 greedy and 9.23 beam.
