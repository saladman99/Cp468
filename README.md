# LSTM vs LLM for Abstractive News Summarization

**Task:** CNN/DailyMail news article → short abstractive summary.

This repository is a project starter for the course assignment comparing a classical
LSTM encoder-decoder with attention against a modern LLM on the **same held-out test
examples**.

## 1. Dataset

The project uses **CNN/DailyMail 3.0.0** from Hugging Face. The dataset has official
train, validation, and test partitions. `prepare_data.py` preserves those partitions
and only subsamples independently inside them with seed 42, so test examples never
enter training or validation.

Default project-sized sample:

- 8,000 training examples
- 1,000 validation examples
- 1,000 test examples

The Hugging Face dataset card currently lists **Apache-2.0** on the repository. It
also states specifically that version 1.0.0 was released under Apache-2.0; the news
text itself was written by CNN and Daily Mail journalists. Document this nuance in
the report rather than claiming ownership of the news articles.

Key citations:

- Hermann et al. (2015), *Teaching Machines to Read and Comprehend*.
- See, Liu, and Manning (2017), *Get To The Point: Summarization with Pointer-Generator Networks*.

## 2. Architecture

Main model:

`word embedding → bidirectional LSTM encoder → Bahdanau attention → LSTM decoder → vocabulary projection`

Ablation:

`word embedding → bidirectional LSTM encoder → fixed mean encoder context → LSTM decoder → vocabulary projection`

The ablation removes dynamic attention while keeping the rest of the setup as close
as possible.

## 3. Environment

Python 3.11+ is recommended.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## 4. Prepare the data

```bash
python prepare_data.py --train_n 8000 --val_n 1000 --test_n 1000 --seed 42
```

This creates:

```text
data/train.csv
data/val.csv
data/test.csv
data/DATASET_CARD.md
```

The CSVs contain:

```text
article,summary
```

## 5. Run tests

```bash
pytest -q
```

## 6. Train the LSTM + attention model

```bash
python train.py --config configs/attention.json
```

Important outputs:

```text
outputs/lstm_attention/best_model.pt
outputs/lstm_attention/history.json
outputs/lstm_attention/run_metadata.json
outputs/lstm_attention/src_vocab.json
outputs/lstm_attention/tgt_vocab.json
```

`run_metadata.json` records the trainable parameter count, total training time,
hardware/device, vocabulary sizes, and best validation loss.

## 7. Train the no-attention ablation

```bash
python train.py --config configs/no_attention.json
```

## 8. Generate LSTM test summaries

Attention model:

```bash
python generate.py --checkpoint outputs/lstm_attention/best_model.pt --data data/test.csv --output outputs/lstm_attention_test.csv
```

Ablation:

```bash
python generate.py --checkpoint outputs/lstm_no_attention/best_model.pt --data data/test.csv --output outputs/lstm_no_attention_test.csv
```

## 9. Evaluate with ROUGE

```bash
python evaluate.py --predictions outputs/lstm_attention_test.csv --output outputs/lstm_attention_metrics.json
python evaluate.py --predictions outputs/lstm_no_attention_test.csv --output outputs/lstm_no_attention_metrics.json
```

Primary metrics:

- ROUGE-1 F1
- ROUGE-2 F1
- ROUGE-L F1

The script also reports results by article-length bucket and simple diagnostic rates
for empty outputs, exact source copying, and repeated bigrams.

## 10. LLM baseline

The script supports all four combinations required for a stronger comparison:

- zero-shot + concise prompt
- zero-shot + faithful prompt
- 4-shot + concise prompt
- 4-shot + faithful prompt

The few-shot demonstrations come **only from `data/train.csv`**. The LLM receives
the same 400-token truncated article representation used by the LSTM.

Set your API key:

```powershell
$env:OPENAI_API_KEY="YOUR_KEY"
```

Run the four baselines:

```bash
python run_llm_baseline.py --setting zero_shot --prompt_variant concise --output outputs/llm_zero_concise.csv
python run_llm_baseline.py --setting zero_shot --prompt_variant faithful --output outputs/llm_zero_faithful.csv
python run_llm_baseline.py --setting few_shot --prompt_variant concise --k 4 --output outputs/llm_four_concise.csv
python run_llm_baseline.py --setting few_shot --prompt_variant faithful --k 4 --output outputs/llm_four_faithful.csv
```

The default model is the pinned snapshot `gpt-5-mini-2025-08-07`. The script records
actual input/output token counts, runtime, prompt text, demonstration examples, and
estimated API cost. The default standard pricing constants are $0.25 per million
input tokens and $2.00 per million output tokens; verify current pricing immediately
before the final experiment and override the CLI values if needed.

Evaluate each output with the same `evaluate.py` command used for the LSTM.

## 11. Compare systems and create the qualitative table

```bash
python analyze_results.py \
  --system LSTM_Attention=outputs/lstm_attention_test.csv \
  --system LSTM_NoAttention=outputs/lstm_no_attention_test.csv \
  --system LLM_Zero_Concise=outputs/llm_zero_concise.csv \
  --system LLM_Zero_Faithful=outputs/llm_zero_faithful.csv \
  --system LLM_4Shot_Concise=outputs/llm_four_concise.csv \
  --system LLM_4Shot_Faithful=outputs/llm_four_faithful.csv
```

This creates:

```text
outputs/system_comparison.csv
outputs/qualitative_examples.csv
```

The second file deliberately spreads 10 examples through the test set instead of
selecting only the best-looking cases. Manually inspect them and assign meaningful
error categories such as:

- missing key information / under-summarization
- repetition
- hallucinated detail
- incorrect entity or number
- overly extractive copying
- OOV / rare-word failure
- poor fluency
- excessive length

You can replace examples if necessary to ensure the final 10 illustrate distinct,
verified behaviors, but document the selection rule and do not cherry-pick wins.

## 12. Demo

After training:

```bash
python demo.py --checkpoint outputs/lstm_attention/best_model.pt
```

Paste an article and the model prints its generated summary. This is useful for the
8-minute demonstration video.

## 13. Suggested 8-minute video structure

1. **0:00-0:45** — task, dataset, and goal.
2. **0:45-2:00** — repository and data splits.
3. **2:00-3:30** — encoder, attention, decoder architecture.
4. **3:30-4:30** — training configuration and reproducibility.
5. **4:30-5:30** — live LSTM demo.
6. **5:30-6:30** — LLM prompt variants and fair-input setup.
7. **6:30-7:30** — ROUGE results and side-by-side examples.
8. **7:30-8:00** — trade-offs and limitations.

## 14. Important fairness decision

CNN/DailyMail articles average much longer than a small word-level LSTM can handle
comfortably. The project therefore truncates articles to 400 tokens for the LSTM.
For the comparison, `run_llm_baseline.py` applies the **same 400-token truncation**
before calling the LLM. This prevents the LLM from receiving a substantially richer
input than the LSTM during the main comparison.

A useful limitation to discuss is that truncation may remove information near the end
of articles. You can optionally report a secondary full-article LLM experiment, but
keep it separate from the main apples-to-apples comparison.

## 15. Report

Use `report/report_template.md`. Replace every placeholder with measured values from
your runs. Do **not** invent ROUGE scores, training time, cost, or qualitative claims.

Before submission, add:

- GitHub URL
- video URL
- exact hardware
- exact training time
- model parameter count
- ROUGE table
- four LLM prompt results
- 10 verified qualitative examples
- contribution statement
- AI-use disclosure
