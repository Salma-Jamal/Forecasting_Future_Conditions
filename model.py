import torch
import torch.nn as nn
import torch.nn.functional as F


class EHRDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_types: int = 7,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.15,
        max_seq_len: int = 256,
        n_time_features: int = 3,
        num_genders: int = 3,
        num_races: int = 6,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_time_features = n_time_features

        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.type_embed = nn.Embedding(num_types, d_model)
        self.time_proj = nn.Linear(n_time_features, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.gender_embed = nn.Embedding(num_genders, d_model)
        self.race_embed = nn.Embedding(num_races, d_model)

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        causal_mask = torch.triu(
            torch.full((max_seq_len, max_seq_len), True), diagonal=1
        )
        self.register_buffer("causal_mask", causal_mask)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.lm_head = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.token_embed.weight, mean=0, std=0.02)
        with torch.no_grad():
            self.token_embed.weight[0].zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        type_ids: torch.Tensor,
        time_features: torch.Tensor,
        attention_mask: torch.Tensor,
        gender_id: torch.Tensor = None,
        race_id: torch.Tensor = None,
        return_hidden: bool = False,
    ):
        b, s = input_ids.shape

        token_x = self.token_embed(input_ids)
        type_x = self.type_embed(type_ids)
        if time_features.dim() == 2:
            time_features = time_features.unsqueeze(-1)
        time_x = self.time_proj(time_features)

        positions = torch.arange(s, device=input_ids.device).unsqueeze(0)
        pos_x = self.pos_embed(positions)

        x = token_x + type_x + time_x + pos_x

        if gender_id is not None:
            g = self.gender_embed(gender_id)  # (b, d_model)
            x = x + g.unsqueeze(1)
        if race_id is not None:
            r = self.race_embed(race_id)     # (b, d_model)
            x = x + r.unsqueeze(1)

        x = x * (self.d_model ** 0.5)
        x = self.layer_norm(x)
        x = self.dropout(x)

        causal_mask = self.causal_mask[:s, :s]
        key_padding_mask = (attention_mask == 0)

        x = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
        )

        if return_hidden:
            return x

        logits = self.lm_head(x)
        return logits

    def compute_pretrain_loss(self, logits, input_ids):
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
            ignore_index=0,
        )
        return loss


class EHRDecoderForClassification(nn.Module):
    def __init__(self, decoder: EHRDecoder, num_labels: int,
                 classifier_dropout: float = 0.3, classifier_hidden: int = 256):
        super().__init__()
        self.decoder = decoder
        self.classifier = nn.Sequential(
            nn.LayerNorm(decoder.d_model),
            nn.Linear(decoder.d_model, classifier_hidden),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden, num_labels),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        type_ids: torch.Tensor,
        time_features: torch.Tensor,
        attention_mask: torch.Tensor,
        gender_id: torch.Tensor = None,
        race_id: torch.Tensor = None,
        return_lm_logits: bool = False,
    ):
        hidden = self.decoder(
            input_ids, type_ids, time_features, attention_mask,
            gender_id=gender_id, race_id=race_id,
            return_hidden=True,
        )
        # last-token pooling: gather the last non-padded position for each row
        lengths = attention_mask.sum(dim=1).clamp(min=1) - 1  # (b,)
        idx = lengths.view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
        pooled = hidden.gather(1, idx).squeeze(1)  # (b, d_model)
        logits = self.classifier(pooled)

        if return_lm_logits:
            lm_logits = self.decoder.lm_head(hidden)
            return logits, lm_logits

        return logits
