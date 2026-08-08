"""
Dataset for abstractive summarization (CNN/DailyMail: article -> highlights).

Expects a CSV with two columns: "article" and "summary".
The official train/validation/test split must be created upstream before this
class is used. Vocabulary construction must use training data only.
"""

import csv

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from vocab import detokenize, tokenize


class SummaryDataset(Dataset):
    def __init__(self, csv_path, src_vocab, tgt_vocab, max_src_len=400, max_tgt_len=80):
        self.pairs = []
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                article_tokens = tokenize(row["article"])[:max_src_len]
                summary_tokens = tokenize(row["summary"])[: max_tgt_len - 2]
                if article_tokens and summary_tokens:
                    self.pairs.append((article_tokens, summary_tokens))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_tokens, tgt_tokens = self.pairs[idx]
        src_ids = self.src_vocab.encode(src_tokens, add_sos_eos=False)
        tgt_ids = self.tgt_vocab.encode(tgt_tokens, add_sos_eos=True)
        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long),
            "article": detokenize(src_tokens),
            "summary": detokenize(tgt_tokens),
        }


def make_collate_fn(src_pad_id, tgt_pad_id):
    def collate_fn(batch):
        src_seqs = [item["src_ids"] for item in batch]
        tgt_seqs = [item["tgt_ids"] for item in batch]

        src_lengths = torch.tensor([len(s) for s in src_seqs], dtype=torch.long)
        tgt_lengths = torch.tensor([len(t) for t in tgt_seqs], dtype=torch.long)

        src_padded = pad_sequence(src_seqs, batch_first=True, padding_value=src_pad_id)
        tgt_padded = pad_sequence(tgt_seqs, batch_first=True, padding_value=tgt_pad_id)

        return {
            "src": src_padded,
            "src_lengths": src_lengths,
            "tgt": tgt_padded,
            "tgt_lengths": tgt_lengths,
            "articles": [item["article"] for item in batch],
            "summaries": [item["summary"] for item in batch],
        }

    return collate_fn
