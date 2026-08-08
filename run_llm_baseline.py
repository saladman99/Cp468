import argparse
import csv
import json
import os
import random
import time

from openai import OpenAI
from tqdm import tqdm

from vocab import detokenize, tokenize


PROMPTS = {
    "concise": (
        "Summarize the news article in 2-3 concise sentences. Keep only the most "
        "important information. Do not add facts that are not supported by the article. "
        "Output only the summary."
    ),
    "faithful": (
        "Write a faithful abstractive summary of the news article. Preserve the key "
        "people, events, and outcome, remove secondary details, and do not speculate. "
        "Keep the summary under 80 tokens. Output only the summary."
    ),
}


def truncate(text, max_tokens):
    return detokenize(tokenize(text)[:max_tokens])


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def choose_examples(train_rows, k, seed, max_src_len, max_tgt_len):
    rng = random.Random(seed)
    candidates = train_rows.copy()
    rng.shuffle(candidates)
    examples = []
    for row in candidates:
        article = truncate(row["article"], max_src_len)
        summary = truncate(row["summary"], max_tgt_len - 2)
        if article and summary:
            examples.append((article, summary))
        if len(examples) == k:
            break
    return examples


def make_prompt(article, instruction, examples):
    parts = [instruction]
    for i, (ex_article, ex_summary) in enumerate(examples, 1):
        parts.append(
            f"\nExample {i}\nARTICLE:\n{ex_article}\nSUMMARY:\n{ex_summary}"
        )
    parts.append(f"\nARTICLE:\n{article}\nSUMMARY:\n")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", default="data/test.csv")
    parser.add_argument("--train_file", default="data/train.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="gpt-5-mini-2025-08-07")
    parser.add_argument("--setting", choices=["zero_shot", "few_shot"], required=True)
    parser.add_argument("--prompt_variant", choices=list(PROMPTS), required=True)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_src_len", type=int, default=400)
    parser.add_argument("--max_tgt_len", type=int, default=80)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--input_cost_per_million", type=float, default=0.25)
    parser.add_argument("--output_cost_per_million", type=float, default=2.00)
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY before running the LLM baseline.")

    client = OpenAI()
    test_rows = load_rows(args.test_file)
    if args.limit:
        test_rows = test_rows[:args.limit]

    examples = []
    if args.setting == "few_shot":
        train_rows = load_rows(args.train_file)
        examples = choose_examples(
            train_rows, args.k, args.seed, args.max_src_len, args.max_tgt_len
        )

    total_input_tokens = 0
    total_output_tokens = 0
    total_seconds = 0.0

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "article", "reference", "prediction", "model", "setting",
            "prompt_variant", "input_tokens", "output_tokens", "seconds",
            "response_id"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in tqdm(test_rows, desc=f"{args.setting}/{args.prompt_variant}"):
            article = truncate(row["article"], args.max_src_len)
            reference = truncate(row["summary"], args.max_tgt_len - 2)
            prompt = make_prompt(article, PROMPTS[args.prompt_variant], examples)

            start = time.perf_counter()
            response = client.responses.create(
                model=args.model,
                input=prompt,
            )
            elapsed = time.perf_counter() - start
            prediction = response.output_text.strip()

            usage = getattr(response, "usage", None)
            input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
            output_tokens = getattr(usage, "output_tokens", 0) if usage else 0

            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            total_seconds += elapsed

            writer.writerow({
                "article": article,
                "reference": reference,
                "prediction": prediction,
                "model": args.model,
                "setting": args.setting,
                "prompt_variant": args.prompt_variant,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "seconds": elapsed,
                "response_id": response.id,
            })
            f.flush()

    estimated_cost = (
        total_input_tokens / 1_000_000 * args.input_cost_per_million
        + total_output_tokens / 1_000_000 * args.output_cost_per_million
    )
    metadata = {
        "model": args.model,
        "setting": args.setting,
        "prompt_variant": args.prompt_variant,
        "instruction": PROMPTS[args.prompt_variant],
        "few_shot_examples": examples,
        "test_examples": len(test_rows),
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "runtime_seconds": total_seconds,
        "input_cost_per_million_usd": args.input_cost_per_million,
        "output_cost_per_million_usd": args.output_cost_per_million,
        "estimated_cost_usd": estimated_cost,
    }
    metadata_path = args.output.rsplit(".", 1)[0] + "_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
