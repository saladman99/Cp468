import argparse
import csv

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SummaryDataset, make_collate_fn
from model import Seq2SeqSummarizer
from utils import get_device
from vocab import Vocab, detokenize


def model_from_checkpoint(checkpoint, device):
    config = checkpoint["config"]
    src_vocab = Vocab(checkpoint["src_vocab"])
    tgt_vocab = Vocab(checkpoint["tgt_vocab"])

    model = Seq2SeqSummarizer(
        len(src_vocab),
        len(tgt_vocab),
        config["embed_dim"],
        config["encoder_hidden_dim"],
        config["decoder_hidden_dim"],
        src_vocab.pad_id,
        tgt_vocab.pad_id,
        config["dropout"],
        config["use_attention"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, src_vocab, tgt_vocab, config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="data/test.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = get_device()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model, src_vocab, tgt_vocab, config = model_from_checkpoint(checkpoint, device)

    dataset = SummaryDataset(
        args.data, src_vocab, tgt_vocab,
        config["max_src_len"], config["max_tgt_len"]
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(src_vocab.pad_id, tgt_vocab.pad_id),
    )

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["article", "reference", "prediction"])
        writer.writeheader()

        for batch in tqdm(loader, desc="generating"):
            src = batch["src"].to(device)
            generated = model.greedy_decode(
                src,
                batch["src_lengths"],
                tgt_vocab.sos_id,
                tgt_vocab.eos_id,
                max_len=config["max_tgt_len"],
            )

            for article, reference, ids in zip(
                batch["articles"], batch["summaries"], generated.cpu().tolist()
            ):
                prediction = detokenize(tgt_vocab.decode(ids))
                writer.writerow({
                    "article": article,
                    "reference": reference,
                    "prediction": prediction,
                })

    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
