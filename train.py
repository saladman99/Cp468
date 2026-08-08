import argparse
import csv
import json
import platform
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SummaryDataset, make_collate_fn
from model import Seq2SeqSummarizer
from utils import count_parameters, get_device, load_json, save_json, set_seed
from vocab import Vocab, tokenize


def article_token_sequences(csv_path, max_len):
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            yield tokenize(row["article"])[:max_len]


def summary_token_sequences(csv_path, max_len):
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            yield tokenize(row["summary"])[: max_len - 2]


def build_model(config, src_vocab, tgt_vocab):
    return Seq2SeqSummarizer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        embed_dim=config["embed_dim"],
        encoder_hidden_dim=config["encoder_hidden_dim"],
        decoder_hidden_dim=config["decoder_hidden_dim"],
        src_pad_id=src_vocab.pad_id,
        tgt_pad_id=tgt_vocab.pad_id,
        dropout=config["dropout"],
        use_attention=config["use_attention"],
    )


def run_epoch(model, loader, criterion, device, optimizer, teacher_forcing, clip):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_tokens = 0

    bar = tqdm(loader, desc="train" if training else "valid", leave=False)
    for batch in bar:
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)
        src_lengths = batch["src_lengths"]

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(
                src,
                src_lengths,
                tgt,
                teacher_forcing_ratio=teacher_forcing if training else 1.0,
            )
            gold = tgt[:, 1:]
            loss = criterion(logits.reshape(-1, logits.size(-1)), gold.reshape(-1))

            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

        non_pad = gold.ne(model.tgt_pad_id).sum().item()
        total_loss += loss.item() * non_pad
        total_tokens += non_pad
        bar.set_postfix(loss=f"{total_loss / max(total_tokens, 1):.4f}")

    return total_loss / max(total_tokens, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_json(args.config)
    set_seed(config["seed"])

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(config, output_dir / "config.json")

    src_vocab = Vocab.build(
        article_token_sequences(config["train_file"], config["max_src_len"]),
        max_size=config["src_vocab_size"],
        min_freq=config["min_freq"],
    )
    tgt_vocab = Vocab.build(
        summary_token_sequences(config["train_file"], config["max_tgt_len"]),
        max_size=config["tgt_vocab_size"],
        min_freq=config["min_freq"],
    )
    src_vocab.save(output_dir / "src_vocab.json")
    tgt_vocab.save(output_dir / "tgt_vocab.json")

    train_data = SummaryDataset(
        config["train_file"], src_vocab, tgt_vocab,
        config["max_src_len"], config["max_tgt_len"]
    )
    val_data = SummaryDataset(
        config["val_file"], src_vocab, tgt_vocab,
        config["max_src_len"], config["max_tgt_len"]
    )

    collate = make_collate_fn(src_vocab.pad_id, tgt_vocab.pad_id)
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(
        train_data,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate,
        num_workers=config["num_workers"],
        generator=generator,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate,
        num_workers=config["num_workers"],
    )

    device = get_device()
    model = build_model(config, src_vocab, tgt_vocab).to(device)
    optimizer = Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_id)

    parameter_count = count_parameters(model)
    print(f"Device: {device}")
    print(f"Trainable parameters: {parameter_count:,}")
    print(f"Source vocabulary: {len(src_vocab):,}")
    print(f"Target vocabulary: {len(tgt_vocab):,}")

    best_val = float("inf")
    bad_epochs = 0
    history = []
    training_start = time.perf_counter()

    for epoch in range(1, config["epochs"] + 1):
        epoch_start = time.perf_counter()
        train_loss = run_epoch(
            model, train_loader, criterion, device, optimizer,
            config["teacher_forcing_ratio"], config["gradient_clip"]
        )
        val_loss = run_epoch(
            model, val_loader, criterion, device, None,
            1.0, config["gradient_clip"]
        )
        elapsed = time.perf_counter() - epoch_start

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "seconds": elapsed,
        }
        history.append(row)
        save_json(history, output_dir / "history.json")
        print(json.dumps(row))

        if val_loss < best_val:
            best_val = val_loss
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "src_vocab": src_vocab.token_to_id,
                    "tgt_vocab": tgt_vocab.token_to_id,
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                output_dir / "best_model.pt",
            )
        else:
            bad_epochs += 1
            if bad_epochs >= config["patience"]:
                print("Early stopping")
                break

    total_seconds = time.perf_counter() - training_start
    metadata = {
        "experiment_name": config["experiment_name"],
        "trainable_parameters": parameter_count,
        "training_seconds": total_seconds,
        "best_val_loss": best_val,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "train_examples": len(train_data),
        "val_examples": len(val_data),
        "src_vocab_size": len(src_vocab),
        "tgt_vocab_size": len(tgt_vocab),
    }
    save_json(metadata, output_dir / "run_metadata.json")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
