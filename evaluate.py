import argparse
import csv
import json
from collections import Counter

from rouge_score import rouge_scorer

from vocab import tokenize


def ngram_counts(tokens, n):
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def repeated_bigram_rate(text):
    tokens = tokenize(text)
    bigrams = list(ngram_counts(tokens, 2).elements())
    if not bigrams:
        return 0.0
    unique = len(set(bigrams))
    return 1.0 - unique / len(bigrams)


def length_bucket(article):
    n = len(tokenize(article))
    if n <= 150:
        return "short"
    if n <= 300:
        return "medium"
    return "long"


def score_rows(rows):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    copy_count = 0
    empty_count = 0
    pred_lengths = 0
    repeat_total = 0.0

    for row in rows:
        reference = row["reference"]
        prediction = row["prediction"]
        scores = scorer.score(reference, prediction)
        for key in totals:
            totals[key] += scores[key].fmeasure

        if prediction.strip().lower() == row["article"].strip().lower():
            copy_count += 1
        if not prediction.strip():
            empty_count += 1
        pred_lengths += len(tokenize(prediction))
        repeat_total += repeated_bigram_rate(prediction)

    n = max(len(rows), 1)
    return {
        "count": len(rows),
        "rouge1_f1": 100 * totals["rouge1"] / n,
        "rouge2_f1": 100 * totals["rouge2"] / n,
        "rougeL_f1": 100 * totals["rougeL"] / n,
        "mean_prediction_tokens": pred_lengths / n,
        "source_copy_rate": copy_count / n,
        "empty_output_rate": empty_count / n,
        "mean_repeated_bigram_rate": repeat_total / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.predictions, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    result = {"overall": score_rows(rows), "length_buckets": {}}
    for bucket in ["short", "medium", "long"]:
        subset = [r for r in rows if length_bucket(r["article"]) == bucket]
        result["length_buckets"][bucket] = score_rows(subset) if subset else {"count": 0}

    print(json.dumps(result, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
