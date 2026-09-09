# Phase 1: Data Preparation & Exploration

## 1. Objective & Task Overview
The goal of Phase 1 is to build a clean, reliable, and reproducible data pipeline for **Urdu Question Generation (QG)**.
In question generation, the model is trained to generate a relevant question given a context sentence and an answer.

* **Task Formulation (Seq2Seq):**
  * **Source (Input):** The specific sentence from the context that contains the answer, with the answer span highlighted using custom delimiter tags (`<ans> ... </ans>`).
  * **Target (Output):** The corresponding Urdu question.

---

## 2. Environment & Dependencies
The following core libraries were installed and utilized:
* `datasets` (Hugging Face): For dataset loading and stream processing.
* `torch`: Deep learning framework for downstream modeling.
* `sentencepiece`: Tokenizer backend for multilingual subword models.
* `sacrebleu` & `rouge-score`: Evaluation metrics for downstream generation benchmarks.
* `matplotlib` & `numpy`: Length analysis and distribution histograms.

```bash
pip install datasets sentencepiece sacrebleu rouge-score torch matplotlib numpy
```

---

## 3. Datasets Used & Schema Inspection

### 3.1 Primary Training & Validation: `uqa/UQA`
* **Origin:** Translated Urdu version of Stanford Question Answering Dataset (SQuAD 2.0), aligned using the EATS (Enclose to Anchor, Translate, Seek) methodology.
* **Splits & Counts:**
  * **Train:** 124,745 raw rows (83,018 answerable)
  * **Validation:** 16,824 raw rows
* **Schema Details:**
  * `id`: Unique example identifier
  * `title`: Article title
  * `context`: Full Urdu context paragraph
  * `question`: Target Urdu question
  * `answer`: Plain string containing the answer text
  * `answer_start`: Character start offset of the answer inside the context
  * `is_impossible`: Boolean flag indicating unanswerable questions

> **Key Technical Note:** Unlike standard English SQuAD (which stores `answers` as a nested dictionary `{"text": [...], "answer_start": [...]}`), this dataset stores `answer` directly as a string and `answer_start` as an integer.

### 3.2 Benchmark Out-of-Domain Test: `uqa/Wiki-UQA`
* **Purpose:** Out-of-domain zero-shot evaluation test set.
* **Structure:** 210 human-annotated question-answer pairs derived from Urdu Wikipedia articles.
* Saved directly to `wiki_test.tsv` (not used during model training to avoid data leakage).

---

## 4. Pipeline & Processing Steps

### Step 1: Filtering Non-Empty Answers (Req 1.1)
* Only answerable questions are kept for Question Generation.
* Unanswerable rows (empty `answer` or `is_impossible == True`) are filtered out.

### Step 2: Sentence Segmentation & Offset Alignment (Req 1.2)
* Text is split into sentences using Urdu-specific sentence terminators:
  * `\u06D4` (Urdu full stop `۔`)
  * `\u061F` (Urdu question mark `؟`)
  * `!` (Exclamation mark)
* Sentence generator tracks exact start and end character offsets: `(s, e, sent)`.
* Locates the exact sentence where `s <= a_start < e`.
* **Integrity Validation:** Validates that `sent[rel : rel + len(a_text)] == a_text` to verify that character offsets did not shift during tokenization or whitespace formatting.

### Step 3: Answer Span Highlighting (Req 1.3)
* The isolated sentence is tagged with `<ans>` and `</ans>` around the target answer:
  ```text
  [sentence_prefix] <ans> [answer_text] </ans> [sentence_suffix]
  ```
* Whitespace is cleaned and normalized using `" ".join(text.split())`.
* The tagged sentence becomes `source` and the question becomes `target`.

### Step 4: Token Length Filtering (Req 1.4)
To eliminate truncated sentences, noisy translations, and outlier inputs that exceed model context windows:
* **Max Source Length:** 60 whitespace tokens (words).
* **Max Target Length:** 25 whitespace tokens (words).
* Any pairs exceeding either threshold are discarded.

### Step 5: Export to TSV & Verification (Req 1.5)
Data is saved into clean Tab-Separated Value (`.tsv`) files using `csv.QUOTE_NONE` and `escapechar="\\"` to preserve text formatting:

| Split File | Number of Pairs | Description |
| :--- | :---: | :--- |
| `train.tsv` | **75,067** | Filtered training pairs |
| `valid.tsv` | **10,018** | Validation pairs for checkpoint selection |
| `wiki_test.tsv` | **177** | Benchmark test pairs |

---

## 5. Sample Output Inspection

**Example 1:**
* **Source:** `ہیوسٹن ، ٹیکساس میں پیدا ہوئی اور اس کی پرورش ہوئی ، اس نے بچپن میں مختلف گانے اور رقص کے مقابلوں میں پرفارم کیا ، اور <ans> 1990 کی دہائی کے آخر میں </ans> R&B گرل گروپ ڈسٹنی چائلڈ کے لیڈ گلوکار کی حیثیت سے شہرت حاصل کی۔`
* **Target:** `بیونس نے کب مقبولیت حاصل کرنا شروع کی؟`

**Example 2:**
* **Source:** `ہیوسٹن ، ٹیکساس میں پیدا اور پرورش پائی ، اس نے بچپن میں مختلف <ans> گانے اور رقص </ans> کے مقابلوں میں پرفارم کیا ، اور 1990 کی دہائی کے آخر میں آر اینڈ بی گرل گروپ ڈسٹنی چائلڈ کی لیڈ گلوکارہ کی حیثیت سے شہرت حاصل کی۔`
* **Target:** `جب وہ بڑی ہو رہی تھی تو بیونس نے کن شعبوں میں مقابلہ کیا؟`

---

## 6. Length Statistics & Histograms
* Source token lengths are clustered between 15 and 45 words.
* Target question lengths are clustered between 6 and 14 words.
* Distributions confirm that sequence lengths are well within maximum token limits for subword tokenizers (such as mT5 / mBART / IndicBART).
