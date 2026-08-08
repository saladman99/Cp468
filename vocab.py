import json
import re
from collections import Counter

PAD = "<pad>"
UNK = "<unk>"
SOS = "<sos>"
EOS = "<eos>"
SPECIAL_TOKENS = [PAD, UNK, SOS, EOS]

_TOKEN_PATTERN = re.compile(r"\w+(?:['’]\w+)?|[^\w\s]", re.UNICODE)


def tokenize(text):
    return _TOKEN_PATTERN.findall(text.lower())


def detokenize(tokens):
    text = " ".join(tokens)
    for mark in [".", ",", "!", "?", ";", ":", "%", ")", "]", "}"]:
        text = text.replace(" " + mark, mark)
    for mark in ["(", "[", "{"]:
        text = text.replace(mark + " ", mark)
    text = text.replace(" ' ", "'")
    return text.strip()


class Vocab:
    def __init__(self, token_to_id):
        self.token_to_id = token_to_id
        self.id_to_token = [None] * len(token_to_id)
        for token, idx in token_to_id.items():
            self.id_to_token[idx] = token

    @classmethod
    def build(cls, token_sequences, max_size=30000, min_freq=2):
        counts = Counter()
        for tokens in token_sequences:
            counts.update(tokens)

        token_to_id = {token: i for i, token in enumerate(SPECIAL_TOKENS)}
        for token, count in counts.most_common():
            if count < min_freq:
                break
            if token in token_to_id:
                continue
            if len(token_to_id) >= max_size:
                break
            token_to_id[token] = len(token_to_id)
        return cls(token_to_id)

    def __len__(self):
        return len(self.token_to_id)

    @property
    def pad_id(self):
        return self.token_to_id[PAD]

    @property
    def unk_id(self):
        return self.token_to_id[UNK]

    @property
    def sos_id(self):
        return self.token_to_id[SOS]

    @property
    def eos_id(self):
        return self.token_to_id[EOS]

    def encode(self, tokens, add_sos_eos=False):
        ids = [self.token_to_id.get(t, self.unk_id) for t in tokens]
        if add_sos_eos:
            ids = [self.sos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids, stop_at_eos=True, skip_special=True):
        tokens = []
        for idx in ids:
            token = self.id_to_token[int(idx)]
            if stop_at_eos and token == EOS:
                break
            if skip_special and token in SPECIAL_TOKENS:
                continue
            tokens.append(token)
        return tokens

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))
