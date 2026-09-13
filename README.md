
![Uploading frontend.png…]()

# Urdu Question Generator

An answer-conditioned encoder-decoder that generates an Urdu question from a sentence and a marked answer. The model is a two-layer bidirectional GRU encoder and GRU decoder with Luong attention, trained from scratch without pretrained models or weights.

[Read the project blog](https://medium.com/@musarashid9271/teaching-a-neural-network-to-ask-questions-in-urdu-2b142b39818b?postPublishedType=repub) · [_Read Linkedin Post_](https://www.linkedin.com/posts/muhammad-musa-a97527435_i-along-with-my-teammate-areesha-saqib-built-activity-7504762904299339776-DB77?utm_source=share&utm_medium=member_desktop&rcm=ACoAAG3j41MBsYib5938h681YeTb8DQ3ynZVQTc)

## Results

| Dataset | Greedy BLEU-4 | Beam BLEU-4 |
| :-- | --: | --: |
| UQA validation | 10.16 | **11.29** |
| Wiki-UQA | 8.09 | **9.23** |

Grouped multi-reference BLEU is 11.65 on validation and 9.37 on Wiki-UQA. Full results are in [`results/results.md`](results/results.md).

Training history, dataset statistics, model configuration, validation samples, tokenizer examples, qualitative labels, and discussion are in `results/`. The required loss curve, length histograms, attention heatmap, and current interface screenshot are in `results/figures/`.

## Architecture

```text
Marked sentence
      ↓
SentencePiece tokenizer
      ↓
BiGRU encoder + answer membership and distance features
      ↓
Luong attention + pooled answer representation
      ↓
GRU decoder with input feeding
      ↓
Generated question
```

The revised model has 14.6 million parameters. It uses tied embeddings, a compact output projection, dropout, gradient clipping, AdamW, learning-rate reduction, and early stopping.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`.venv` is local and ignored by Git. Each user should create their own environment.

## Run the application

Place the matching trained files at:

```text
artifacts/answer_gru_v1/best.pt
artifacts/answer_gru_v1/tokenizer.model
```

Then run:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8011
```

Open http://127.0.0.1:8011. The API schema is available at http://127.0.0.1:8011/docs.

## Reproduce the pipeline

Run the numbered notebooks in order, or use the equivalent commands:

```bash
python -m scripts.prepare_data
python -m scripts.train_tokenizer
python -m scripts.train --debug --epochs 2
python -m scripts.train
python -m scripts.evaluation
```

The notebooks clone this repository automatically when opened in Colab and store generated files under `MyDrive/Urdu-QG-v2`. Local commands write regenerated data and model files under ignored `artifacts/`, leaving the committed datasets unchanged.

## Project structure

```text
app/             FastAPI application, model, tokenizer, and inference
frontend/        HTML, CSS, and JavaScript interface
scripts/         Data preparation, training, evaluation, and validation
notebooks/       Colab workflow in execution order
data/            Prepared UQA and Wiki-UQA splits
artifacts/       Local tokenizer and trained checkpoint
results/         Final metrics
config.py        Shared paths and model configuration
requirements.txt Python dependencies
```

Validate the committed data with:

```bash
python -m scripts.validate_data
```
