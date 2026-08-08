"""
Download CNN/DailyMail article-summary pairs and export train/val/test CSVs.

The script keeps the dataset's official train/validation/test partitions, then
subsamples inside each partition with a fixed seed. This avoids split leakage.

Usage:
    pip install datasets
    python prepare_data.py --train_n 8000 --val_n 1000 --test_n 1000

Output:
    data/train.csv
    data/val.csv
    data/test.csv
    data/DATASET_CARD.md
"""

import argparse
import csv
import os

from datasets import load_dataset


def export_split(dataset_split, n, out_path, seed=42):
    n = min(n, len(dataset_split))
    subset = dataset_split.shuffle(seed=seed).select(range(n))

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["article", "summary"])
        written = 0
        for row in subset:
            article = row["article"].strip()
            summary = row["highlights"].strip()
            if article and summary:
                writer.writerow([article, summary])
                written += 1

    print(f"Wrote {written} examples to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_n", type=int, default=8000)
    parser.add_argument("--val_n", type=int, default=1000)
    parser.add_argument("--test_n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="data")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Downloading CNN/DailyMail 3.0.0...")
    dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")

    export_split(dataset["train"], args.train_n,
                 os.path.join(args.out_dir, "train.csv"), args.seed)
    export_split(dataset["validation"], args.val_n,
                 os.path.join(args.out_dir, "val.csv"), args.seed)
    export_split(dataset["test"], args.test_n,
                 os.path.join(args.out_dir, "test.csv"), args.seed)

    card_path = os.path.join(args.out_dir, "DATASET_CARD.md")
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(
            "# CNN/DailyMail dataset provenance\n\n"
            "Source: Hugging Face `abisee/cnn_dailymail`, config `3.0.0`.\n\n"
            "Task: abstractive summarization (`article` -> `highlights`).\n\n"
            "Official partitions are preserved. The script only subsamples "
            "inside each official partition with a fixed random seed.\n\n"
            f"- train sample: {args.train_n}\n"
            f"- validation sample: {args.val_n}\n"
            f"- test sample: {args.test_n}\n"
            f"- seed: {args.seed}\n\n"
            "The Hugging Face dataset repository currently displays the "
            "Apache-2.0 license tag. The dataset card specifically states that "
            "version 1.0.0 was released under Apache-2.0; the underlying news "
            "articles were written by CNN and Daily Mail journalists and remain "
            "copyrighted source material. For the final report, cite the dataset "
            "card and the original papers and describe this licensing nuance.\n\n"
            "Recommended citations:\n"
            "- Hermann et al. (2015), Teaching Machines to Read and Comprehend.\n"
            "- See, Liu, and Manning (2017), Get To The Point: Summarization "
            "with Pointer-Generator Networks.\n"
        )

    print(f"Wrote dataset card to {card_path}")


if __name__ == "__main__":
    main()
