import argparse
import csv
import json

from evaluate import length_bucket, score_rows


def load_predictions(path, name):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return name, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", action="append", required=True,
                        help="NAME=predictions.csv; repeat for each system")
    parser.add_argument("--table", default="outputs/system_comparison.csv")
    parser.add_argument("--examples", default="outputs/qualitative_examples.csv")
    parser.add_argument("--num_examples", type=int, default=10)
    args = parser.parse_args()

    systems = []
    for item in args.system:
        name, path = item.split("=", 1)
        systems.append(load_predictions(path, name))

    summary_rows = []
    for name, rows in systems:
        overall = score_rows(rows)
        summary_rows.append({"system": name, "bucket": "overall", **overall})
        for bucket in ["short", "medium", "long"]:
            subset = [r for r in rows if length_bucket(r["article"]) == bucket]
            if subset:
                summary_rows.append({"system": name, "bucket": bucket, **score_rows(subset)})

    if summary_rows:
        with open(args.table, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)

    common_n = min(len(rows) for _, rows in systems)
    chosen = []
    if common_n:
        # Spread the examples across the test file instead of cherry-picking.
        if args.num_examples == 1:
            indices = [0]
        else:
            indices = [round(i * (common_n - 1) / (args.num_examples - 1))
                       for i in range(args.num_examples)]
        for idx in indices:
            base = systems[0][1][idx]
            record = {
                "index": idx,
                "article": base["article"],
                "reference": base["reference"],
                "length_bucket": length_bucket(base["article"]),
                "error_category": "FILL_MANUALLY",
                "notes": "FILL_MANUALLY",
            }
            for name, rows in systems:
                record[name] = rows[idx]["prediction"]
            chosen.append(record)

    if chosen:
        with open(args.examples, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=chosen[0].keys())
            writer.writeheader()
            writer.writerows(chosen)

    print(json.dumps({
        "comparison_table": args.table,
        "qualitative_examples": args.examples,
        "systems": [name for name, _ in systems],
    }, indent=2))


if __name__ == "__main__":
    main()
