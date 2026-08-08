import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_id, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, src, src_lengths):
        embedded = self.dropout(self.embedding(src))
        packed = pack_padded_sequence(
            embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_outputs, (hidden, cell) = self.lstm(packed)
        outputs, _ = pad_packed_sequence(
            packed_outputs, batch_first=True, total_length=src.size(1)
        )
        return outputs, hidden, cell


class BahdanauAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim):
        super().__init__()
        self.enc_proj = nn.Linear(encoder_dim, decoder_dim, bias=False)
        self.dec_proj = nn.Linear(decoder_dim, decoder_dim, bias=False)
        self.score = nn.Linear(decoder_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, src_mask):
        energy = torch.tanh(
            self.enc_proj(encoder_outputs) + self.dec_proj(decoder_hidden).unsqueeze(1)
        )
        scores = self.score(energy).squeeze(-1)
        scores = scores.masked_fill(~src_mask, -1e9)
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, weights


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        encoder_dim,
        decoder_dim,
        pad_id,
        dropout=0.2,
        use_attention=True,
    ):
        super().__init__()
        self.use_attention = use_attention
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)
        self.attention = (
            BahdanauAttention(encoder_dim, decoder_dim) if use_attention else None
        )
        self.lstm = nn.LSTM(embed_dim + encoder_dim, decoder_dim, batch_first=True)
        self.output = nn.Linear(decoder_dim + encoder_dim + embed_dim, vocab_size)

    def step(self, token, hidden, cell, encoder_outputs, src_mask, fixed_context):
        embedded = self.dropout(self.embedding(token))

        if self.use_attention:
            context, weights = self.attention(hidden[-1], encoder_outputs, src_mask)
        else:
            context = fixed_context
            weights = torch.zeros(
                encoder_outputs.size(0), encoder_outputs.size(1), device=encoder_outputs.device
            )

        decoder_input = torch.cat([embedded, context], dim=-1).unsqueeze(1)
        output, (hidden, cell) = self.lstm(decoder_input, (hidden, cell))
        output = output.squeeze(1)
        logits = self.output(torch.cat([output, context, embedded], dim=-1))
        return logits, hidden, cell, weights


class Seq2SeqSummarizer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        embed_dim,
        encoder_hidden_dim,
        decoder_hidden_dim,
        src_pad_id,
        tgt_pad_id,
        dropout=0.2,
        use_attention=True,
    ):
        super().__init__()
        encoder_dim = encoder_hidden_dim * 2
        self.src_pad_id = src_pad_id
        self.tgt_pad_id = tgt_pad_id

        self.encoder = Encoder(
            src_vocab_size, embed_dim, encoder_hidden_dim, src_pad_id, dropout
        )
        self.hidden_bridge = nn.Linear(encoder_dim, decoder_hidden_dim)
        self.cell_bridge = nn.Linear(encoder_dim, decoder_hidden_dim)
        self.decoder = Decoder(
            tgt_vocab_size,
            embed_dim,
            encoder_dim,
            decoder_hidden_dim,
            tgt_pad_id,
            dropout,
            use_attention,
        )

    def encode(self, src, src_lengths):
        encoder_outputs, hidden, cell = self.encoder(src, src_lengths)

        final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        final_cell = torch.cat([cell[-2], cell[-1]], dim=-1)
        dec_hidden = torch.tanh(self.hidden_bridge(final_hidden)).unsqueeze(0)
        dec_cell = torch.tanh(self.cell_bridge(final_cell)).unsqueeze(0)

        src_mask = src.ne(self.src_pad_id)
        mask_float = src_mask.unsqueeze(-1).float()
        fixed_context = (encoder_outputs * mask_float).sum(dim=1)
        fixed_context = fixed_context / mask_float.sum(dim=1).clamp_min(1.0)

        return encoder_outputs, dec_hidden, dec_cell, src_mask, fixed_context

    def forward(self, src, src_lengths, tgt, teacher_forcing_ratio=0.5):
        batch_size, tgt_len = tgt.shape
        vocab_size = self.decoder.output.out_features
        logits = torch.zeros(batch_size, tgt_len - 1, vocab_size, device=src.device)

        encoder_outputs, hidden, cell, src_mask, fixed_context = self.encode(
            src, src_lengths
        )
        input_token = tgt[:, 0]

        for step in range(tgt_len - 1):
            step_logits, hidden, cell, _ = self.decoder.step(
                input_token, hidden, cell, encoder_outputs, src_mask, fixed_context
            )
            logits[:, step] = step_logits
            predicted = step_logits.argmax(dim=-1)

            use_teacher = torch.rand(1, device=src.device).item() < teacher_forcing_ratio
            input_token = tgt[:, step + 1] if use_teacher else predicted

        return logits

    @torch.no_grad()
    def greedy_decode(self, src, src_lengths, sos_id, eos_id, max_len=80):
        encoder_outputs, hidden, cell, src_mask, fixed_context = self.encode(
            src, src_lengths
        )
        input_token = torch.full(
            (src.size(0),), sos_id, dtype=torch.long, device=src.device
        )

        generated = []
        finished = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)

        for _ in range(max_len):
            logits, hidden, cell, _ = self.decoder.step(
                input_token, hidden, cell, encoder_outputs, src_mask, fixed_context
            )
            input_token = logits.argmax(dim=-1)
            generated.append(input_token)
            finished |= input_token.eq(eos_id)
            if finished.all():
                break

        if not generated:
            return torch.empty(src.size(0), 0, dtype=torch.long, device=src.device)
        return torch.stack(generated, dim=1)
