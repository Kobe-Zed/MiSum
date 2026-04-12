from __future__ import annotations

import math

import torch
from torch import nn

from misum_repro.data import EDGE_TYPES, INTENTS, NODE_TYPE_NAMES


def scatter_softmax(scores: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    if scores.numel() == 0:
        return scores

    heads = scores.size(1)
    outputs = torch.zeros_like(scores)
    for head in range(heads):
        flat_scores = scores[:, head]
        max_values = torch.full((dim_size,), float("-inf"), device=scores.device, dtype=flat_scores.dtype)
        max_values.scatter_reduce_(0, index, flat_scores, reduce="amax", include_self=True)
        shifted = torch.exp(flat_scores - max_values[index])
        denom = torch.zeros(dim_size, device=scores.device, dtype=shifted.dtype)
        denom.index_add_(0, index, shifted)
        outputs[:, head] = shifted / denom[index].clamp_min(1e-9)
    return outputs


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, d_model: int, max_positions: int) -> None:
        super().__init__()
        position = torch.arange(max_positions, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_positions, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        clipped = positions.clamp_max(self.pe.size(0) - 1)
        return self.pe[clipped]


def apply_type_linears(linears: nn.ModuleList, x: torch.Tensor, type_ids: torch.Tensor) -> torch.Tensor:
    output: torch.Tensor | None = None
    for type_index, linear in enumerate(linears):
        mask = type_ids == type_index
        if mask.any():
            projected = linear(x[mask])
            if output is None:
                output = torch.zeros(
                    (x.size(0), projected.size(-1)),
                    device=projected.device,
                    dtype=projected.dtype,
                )
            output[mask] = projected

    if output is None:
        return torch.zeros(
            (x.size(0), linears[0].out_features),
            device=x.device,
            dtype=x.dtype,
        )
    return output


class NodeTextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_node_types: int,
        pad_id: int,
        max_positions: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.node_type_embedding = nn.Embedding(num_node_types, d_model)
        self.position_encoder = SinusoidalPositionEncoding(d_model=d_model, max_positions=max_positions)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_token_ids: torch.Tensor,
        node_type_ids: torch.Tensor,
        node_positions: torch.Tensor,
    ) -> torch.Tensor:
        token_mask = node_token_ids.ne(self.pad_id)
        token_embeddings = self.token_embedding(node_token_ids)
        token_sum = token_embeddings.sum(dim=1)
        token_count = token_mask.sum(dim=1, keepdim=True).clamp_min(1)
        node_text_repr = token_sum / token_count
        node_repr = (
            node_text_repr
            + self.node_type_embedding(node_type_ids)
            + self.position_encoder(node_positions)
        )
        return self.dropout(self.layer_norm(node_repr))


class HeteroGraphAttentionLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_node_types: int,
        num_edge_types: int,
        ff_multiplier: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_node_types)])
        self.k_linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_node_types)])
        self.m_linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_node_types)])
        self.out_linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_node_types)])

        self.att_rel = nn.Parameter(torch.randn(num_edge_types, num_heads, self.head_dim, self.head_dim) * 0.02)
        self.msg_rel = nn.Parameter(torch.randn(num_edge_types, num_heads, self.head_dim, self.head_dim) * 0.02)
        self.rel_prior = nn.Parameter(torch.ones(num_edge_types, num_heads))

        self.attn_dropout = nn.Dropout(dropout)
        self.residual_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_multiplier, d_model),
            nn.Dropout(dropout),
        )
        self.ff_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        node_type_ids: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        if edge_src.numel() == 0:
            return self.ff_norm(x + self.ff(x))

        queries = apply_type_linears(self.q_linears, x, node_type_ids).view(-1, self.num_heads, self.head_dim)
        keys = apply_type_linears(self.k_linears, x, node_type_ids).view(-1, self.num_heads, self.head_dim)
        messages = apply_type_linears(self.m_linears, x, node_type_ids).view(-1, self.num_heads, self.head_dim)

        q_dst = queries[edge_dst]
        k_src = keys[edge_src]
        m_src = messages[edge_src]

        k_rel = torch.einsum("ehd,ehdf->ehf", k_src, self.att_rel[edge_type_ids])
        attn_scores = (q_dst * k_rel).sum(dim=-1) * self.rel_prior[edge_type_ids]
        attn_scores = attn_scores / math.sqrt(self.head_dim)
        attn_weights = scatter_softmax(attn_scores, edge_dst, x.size(0))
        attn_weights = self.attn_dropout(attn_weights)

        rel_messages = torch.einsum("ehd,ehdf->ehf", m_src, self.msg_rel[edge_type_ids])
        weighted_messages = rel_messages * attn_weights.unsqueeze(-1)

        flat_messages = weighted_messages.reshape(weighted_messages.size(0), -1)
        aggregated = torch.zeros(
            (x.size(0), self.num_heads * self.head_dim),
            device=flat_messages.device,
            dtype=flat_messages.dtype,
        )
        aggregated.index_add_(0, edge_dst, flat_messages)
        updated = apply_type_linears(self.out_linears, aggregated, node_type_ids)
        x = self.residual_norm(x + updated)
        return self.ff_norm(x + self.ff(x))


class HeteroGraphEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        num_node_types: int,
        num_edge_types: int,
        ff_multiplier: int,
        pad_id: int,
        max_positions: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.node_encoder = NodeTextEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_node_types=num_node_types,
            pad_id=pad_id,
            max_positions=max_positions,
            dropout=dropout,
        )
        self.layers = nn.ModuleList(
            [
                HeteroGraphAttentionLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    num_node_types=num_node_types,
                    num_edge_types=num_edge_types,
                    ff_multiplier=ff_multiplier,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        node_token_ids: torch.Tensor,
        node_type_ids: torch.Tensor,
        node_positions: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        x = self.node_encoder(node_token_ids, node_type_ids, node_positions)
        for layer in self.layers:
            x = layer(x, node_type_ids, edge_src, edge_dst, edge_type_ids)
        return x


class IntentAwareDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_intents: int,
        pad_id: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.target_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.intent_embedding = nn.Embedding(num_intents, d_model)
        self.decoder = nn.GRU(input_size=d_model * 2, hidden_size=d_model, batch_first=True)
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.intent_proj = nn.Linear(d_model, d_model)
        self.cross_proj = nn.Linear(d_model * 2, d_model)
        self.gate_proj = nn.Linear(d_model * 2, d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def _compute_context(
        self,
        decoder_states: torch.Tensor,
        encoder_states: torch.Tensor,
        encoder_mask: torch.Tensor,
        intent_ids: torch.Tensor,
    ) -> torch.Tensor:
        query = self.query_proj(decoder_states)
        intent_bias = self.intent_proj(self.intent_embedding(intent_ids)).unsqueeze(1).unsqueeze(1)
        key = self.key_proj(encoder_states).unsqueeze(1) + intent_bias
        scores = torch.matmul(query.unsqueeze(2), key.transpose(-1, -2)).squeeze(2)
        scores = scores / math.sqrt(decoder_states.size(-1))
        scores = scores.masked_fill(~encoder_mask.unsqueeze(1), float("-inf"))
        attention = torch.softmax(scores, dim=-1)
        return torch.matmul(attention, encoder_states)

    def forward(
        self,
        encoder_states: torch.Tensor,
        encoder_mask: torch.Tensor,
        intent_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        intent_repr = self.intent_embedding(intent_ids).unsqueeze(1).expand(-1, decoder_input_ids.size(1), -1)
        decoder_inputs = torch.cat([self.target_embedding(decoder_input_ids), intent_repr], dim=-1)

        pooled_encoder = (encoder_states * encoder_mask.unsqueeze(-1)).sum(dim=1)
        pooled_encoder = pooled_encoder / encoder_mask.sum(dim=1, keepdim=True).clamp_min(1)
        decoder_states, _ = self.decoder(decoder_inputs, pooled_encoder.unsqueeze(0))

        iaam_context = self._compute_context(decoder_states, encoder_states, encoder_mask, intent_ids)
        cross_context = torch.tanh(self.cross_proj(torch.cat([decoder_states, iaam_context], dim=-1)))
        gate = torch.sigmoid(self.gate_proj(torch.cat([cross_context, decoder_states], dim=-1)))
        fused_states = gate * cross_context + (1.0 - gate) * decoder_states
        return self.output_proj(self.dropout(fused_states))

    def greedy_decode(
        self,
        encoder_states: torch.Tensor,
        encoder_mask: torch.Tensor,
        intent_ids: torch.Tensor,
        bos_id: int,
        max_new_tokens: int,
    ) -> torch.Tensor:
        batch_size = encoder_states.size(0)
        pooled_encoder = (encoder_states * encoder_mask.unsqueeze(-1)).sum(dim=1)
        pooled_encoder = pooled_encoder / encoder_mask.sum(dim=1, keepdim=True).clamp_min(1)
        hidden = pooled_encoder.unsqueeze(0)

        previous = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=encoder_states.device)
        generated: list[torch.Tensor] = []

        for _ in range(max_new_tokens):
            intent_repr = self.intent_embedding(intent_ids).unsqueeze(1)
            decoder_input = torch.cat([self.target_embedding(previous), intent_repr], dim=-1)
            decoder_state, hidden = self.decoder(decoder_input, hidden)
            context = self._compute_context(decoder_state, encoder_states, encoder_mask, intent_ids)
            cross_context = torch.tanh(self.cross_proj(torch.cat([decoder_state, context], dim=-1)))
            gate = torch.sigmoid(self.gate_proj(torch.cat([cross_context, decoder_state], dim=-1)))
            fused_state = gate * cross_context + (1.0 - gate) * decoder_state
            logits = self.output_proj(fused_state).squeeze(1)
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated.append(next_token)
            previous = next_token

        if generated:
            return torch.cat(generated, dim=1)
        return torch.empty((batch_size, 0), dtype=torch.long, device=encoder_states.device)


class MiSumReproductionModel(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_pad_id: int,
        tgt_pad_id: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        ff_multiplier: int,
        max_positions: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = HeteroGraphEncoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            num_node_types=len(NODE_TYPE_NAMES),
            num_edge_types=len(EDGE_TYPES),
            ff_multiplier=ff_multiplier,
            pad_id=src_pad_id,
            max_positions=max_positions,
            dropout=dropout,
        )
        self.decoder = IntentAwareDecoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_intents=len(INTENTS),
            pad_id=tgt_pad_id,
            dropout=dropout,
        )

    @staticmethod
    def pad_encoder_states(node_states: torch.Tensor, graph_sizes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = graph_sizes.size(0)
        max_nodes = int(graph_sizes.max().item())
        encoder_states = node_states.new_zeros((batch_size, max_nodes, node_states.size(-1)))
        encoder_mask = torch.zeros((batch_size, max_nodes), dtype=torch.bool, device=node_states.device)
        cursor = 0
        for graph_index, size in enumerate(graph_sizes.tolist()):
            encoder_states[graph_index, :size] = node_states[cursor : cursor + size]
            encoder_mask[graph_index, :size] = True
            cursor += size
        return encoder_states, encoder_mask

    def encode_batch(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        node_states = self.encoder(
            node_token_ids=batch["node_token_ids"],
            node_type_ids=batch["node_type_ids"],
            node_positions=batch["node_positions"],
            edge_src=batch["edge_src"],
            edge_dst=batch["edge_dst"],
            edge_type_ids=batch["edge_types"],
        )
        return self.pad_encoder_states(node_states, batch["graph_sizes"])

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        encoder_states, encoder_mask = self.encode_batch(batch)
        decoder_input_ids = batch["targets"][:, :-1]
        return self.decoder(
            encoder_states=encoder_states,
            encoder_mask=encoder_mask,
            intent_ids=batch["intent_ids"],
            decoder_input_ids=decoder_input_ids,
        )

    def generate(
        self,
        batch: dict[str, torch.Tensor],
        bos_id: int,
        max_new_tokens: int,
    ) -> torch.Tensor:
        encoder_states, encoder_mask = self.encode_batch(batch)
        return self.decoder.greedy_decode(
            encoder_states=encoder_states,
            encoder_mask=encoder_mask,
            intent_ids=batch["intent_ids"],
            bos_id=bos_id,
            max_new_tokens=max_new_tokens,
        )
