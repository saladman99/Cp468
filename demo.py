import argparse

import torch

from generate import model_from_checkpoint
from utils import get_device
from vocab import detokenize, tokenize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/lstm_attention/best_model.pt")
    parser.add_argument("--article", default=None)
    args = parser.parse_args()

    device = get_device()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model, src_vocab, tgt_vocab, config = model_from_checkpoint(checkpoint, device)

    article = args.article or input("Paste a news article: ").strip()
    tokens = tokenize(article)[:config["max_src_len"]]
    ids = torch.tensor([src_vocab.encode(tokens)], dtype=torch.long, device=device)
    lengths = torch.tensor([len(tokens)], dtype=torch.long)

    generated = model.greedy_decode(
        ids, lengths, tgt_vocab.sos_id, tgt_vocab.eos_id, config["max_tgt_len"]
    )[0].cpu().tolist()
    summary = detokenize(tgt_vocab.decode(generated))

    print("\nGenerated summary:\n")
    print(summary)


if __name__ == "__main__":
    main()
