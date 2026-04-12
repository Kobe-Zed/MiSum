from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch.utils.data import Dataset

from misum_repro.tokenization import tokenize
from misum_repro.vocab import Vocab


PSEUDO_FUNCTION = 0
PSEUDO_STATEMENT = 1
ASM_BLOCK = 2

NODE_TYPE_NAMES = {
    PSEUDO_FUNCTION: "pseudo_function",
    PSEUDO_STATEMENT: "pseudo_statement",
    ASM_BLOCK: "asm_block",
}

EDGE_TYPES = {
    "PSEUDO_AST": 0,
    "PSEUDO_CFG": 1,
    "PSEUDO_DFG": 2,
    "PSEUDO_NCS": 3,
    "ASM_CFG": 4,
    "ALIGN": 5,
}

INTENTS = ["What", "Why", "How-to-use", "How-it-is-done", "Property", "Others"]
INTENT_TO_ID = {intent: index for index, intent in enumerate(INTENTS)}


@dataclass
class GraphSample:
    sample_id: str
    node_token_ids: list[list[int]]
    node_type_ids: list[int]
    node_positions: list[int]
    edge_src: list[int]
    edge_dst: list[int]
    edge_type_ids: list[int]
    target_ids: list[int]
    target_tokens: list[str]
    intent_id: int
    reference_summary: str


def resolve_split_path(dataset_root: str | Path, split_name: str) -> Path:
    root = Path(dataset_root)
    direct_path = root / f"{split_name}.jsonl"
    if direct_path.exists():
        return direct_path
    if split_name == "valid":
        fallback_path = root / "misum_like_valid.jsonl"
        if fallback_path.exists():
            return fallback_path
    raise FileNotFoundError(f"Unable to find split '{split_name}' under {root}")


def load_raw_split(dataset_root: str | Path, split_name: str, limit: int | None = None) -> list[dict[str, Any]]:
    split_path = resolve_split_path(dataset_root, split_name)
    records: list[dict[str, Any]] = []
    with split_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            records.append(json.loads(line))
    return records


def iter_node_texts(record: dict[str, Any]) -> Iterable[str]:
    for node in sorted(record["pseudo_aast"]["nodes"], key=lambda item: item["id"]):
        identifiers = " ".join(node.get("identifiers", []))
        calls = " ".join(node.get("calls", []))
        constants = " ".join(node.get("constants", []))
        yield f"{node.get('text', '')} {identifiers} {calls} {constants}".strip()
    for block in sorted(record["asm_cfg"]["blocks"], key=lambda item: item["id"]):
        instructions = " ".join(block.get("instructions", []))
        mnemonics = " ".join(block.get("mnemonics", []))
        calls = " ".join(block.get("calls", []))
        constants = " ".join(block.get("constants", []))
        yield f"{block.get('text', '')} {instructions} {mnemonics} {calls} {constants}".strip()


def build_source_corpus(records: Sequence[dict[str, Any]]) -> Iterable[list[str]]:
    for record in records:
        for text in iter_node_texts(record):
            yield tokenize(text)


def build_target_corpus(records: Sequence[dict[str, Any]]) -> Iterable[list[str]]:
    for record in records:
        summary = record["label"]["summary_target"]
        yield tokenize(summary)


def _build_pseudo_node_text(node: dict[str, Any]) -> str:
    identifiers = " ".join(node.get("identifiers", []))
    calls = " ".join(node.get("calls", []))
    constants = " ".join(str(item) for item in node.get("constants", []))
    return f"{node.get('text', '')} {identifiers} {calls} {constants}".strip()


def _build_asm_block_text(block: dict[str, Any]) -> str:
    parts = [
        block.get("text", ""),
        " ".join(block.get("instructions", [])),
        " ".join(block.get("mnemonics", [])),
        " ".join(block.get("calls", [])),
        " ".join(str(item) for item in block.get("constants", [])),
    ]
    return " ".join(part for part in parts if part).strip()


def _heuristic_alignment_edges(pseudo_count: int, asm_count: int, asm_offset: int) -> list[tuple[int, int]]:
    if pseudo_count == 0 or asm_count == 0:
        return []

    edges: list[tuple[int, int]] = []
    for asm_index in range(asm_count):
        if asm_count == 1:
            pseudo_index = 0
        else:
            ratio = asm_index / max(1, asm_count - 1)
            pseudo_index = min(pseudo_count - 1, int(round(ratio * max(1, pseudo_count - 1))))
        asm_node_index = asm_offset + asm_index
        edges.append((pseudo_index, asm_node_index))
        edges.append((asm_node_index, pseudo_index))
    return edges


def preprocess_record(
    record: dict[str, Any],
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    max_pseudo_nodes: int,
    max_asm_blocks: int,
    max_node_tokens: int,
    max_target_tokens: int,
) -> GraphSample:
    pseudo_nodes = sorted(record["pseudo_aast"]["nodes"], key=lambda item: item["id"])[:max_pseudo_nodes]
    asm_blocks = sorted(record["asm_cfg"]["blocks"], key=lambda item: item["id"])[:max_asm_blocks]

    node_token_ids: list[list[int]] = []
    node_type_ids: list[int] = []
    node_positions: list[int] = []

    pseudo_id_map: dict[int, int] = {}
    asm_id_map: dict[int, int] = {}

    for index, node in enumerate(pseudo_nodes):
        pseudo_id_map[node["id"]] = index
        node_positions.append(index)
        node_type_ids.append(PSEUDO_FUNCTION if node.get("kind") == "function" else PSEUDO_STATEMENT)
        node_tokens = tokenize(_build_pseudo_node_text(node))
        node_token_ids.append(src_vocab.encode(node_tokens, max_length=max_node_tokens))

    asm_offset = len(pseudo_nodes)
    for local_index, block in enumerate(asm_blocks):
        asm_id_map[block["id"]] = asm_offset + local_index
        node_positions.append(asm_offset + local_index)
        node_type_ids.append(ASM_BLOCK)
        block_tokens = tokenize(_build_asm_block_text(block))
        node_token_ids.append(src_vocab.encode(block_tokens, max_length=max_node_tokens))

    edge_src: list[int] = []
    edge_dst: list[int] = []
    edge_type_ids: list[int] = []

    for edge in record["pseudo_aast"]["edges"]:
        if edge["src"] in pseudo_id_map and edge["dst"] in pseudo_id_map:
            edge_name = f"PSEUDO_{edge['edge_type']}"
            if edge_name in EDGE_TYPES:
                edge_src.append(pseudo_id_map[edge["src"]])
                edge_dst.append(pseudo_id_map[edge["dst"]])
                edge_type_ids.append(EDGE_TYPES[edge_name])

    for edge in record["asm_cfg"]["edges"]:
        if edge["src"] in asm_id_map and edge["dst"] in asm_id_map:
            edge_src.append(asm_id_map[edge["src"]])
            edge_dst.append(asm_id_map[edge["dst"]])
            edge_type_ids.append(EDGE_TYPES["ASM_CFG"])

    if record.get("alignment_edges"):
        for edge in record["alignment_edges"]:
            pseudo_id = edge.get("pseudo_id")
            asm_id = edge.get("asm_id")
            if pseudo_id in pseudo_id_map and asm_id in asm_id_map:
                edge_src.append(pseudo_id_map[pseudo_id])
                edge_dst.append(asm_id_map[asm_id])
                edge_type_ids.append(EDGE_TYPES["ALIGN"])
    else:
        for src, dst in _heuristic_alignment_edges(len(pseudo_nodes), len(asm_blocks), asm_offset):
            edge_src.append(src)
            edge_dst.append(dst)
            edge_type_ids.append(EDGE_TYPES["ALIGN"])

    summary = record["label"]["summary_target"]
    target_tokens = tokenize(summary)
    target_ids = tgt_vocab.encode(
        target_tokens,
        add_bos=True,
        add_eos=True,
        max_length=max_target_tokens,
    )

    return GraphSample(
        sample_id=record["sample_id"],
        node_token_ids=node_token_ids,
        node_type_ids=node_type_ids,
        node_positions=node_positions,
        edge_src=edge_src,
        edge_dst=edge_dst,
        edge_type_ids=edge_type_ids,
        target_ids=target_ids,
        target_tokens=target_tokens,
        intent_id=INTENT_TO_ID[record["label"]["intent"]],
        reference_summary=summary,
    )


class MiSumDataset(Dataset[GraphSample]):
    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        src_vocab: Vocab,
        tgt_vocab: Vocab,
        max_pseudo_nodes: int,
        max_asm_blocks: int,
        max_node_tokens: int,
        max_target_tokens: int,
    ) -> None:
        self.samples = [
            preprocess_record(
                record=record,
                src_vocab=src_vocab,
                tgt_vocab=tgt_vocab,
                max_pseudo_nodes=max_pseudo_nodes,
                max_asm_blocks=max_asm_blocks,
                max_node_tokens=max_node_tokens,
                max_target_tokens=max_target_tokens,
            )
            for record in records
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> GraphSample:
        return self.samples[index]


def collate_graph_samples(batch: Sequence[GraphSample], src_pad_id: int, tgt_pad_id: int) -> dict[str, Any]:
    batch_size = len(batch)
    graph_sizes = [len(sample.node_type_ids) for sample in batch]
    max_nodes = max(graph_sizes)
    max_node_tokens = max(max(len(tokens) for tokens in sample.node_token_ids) for sample in batch)
    max_target_len = max(len(sample.target_ids) for sample in batch)

    node_token_tensor = torch.full(
        (batch_size, max_nodes, max_node_tokens),
        fill_value=src_pad_id,
        dtype=torch.long,
    )
    node_type_tensor = torch.zeros((batch_size, max_nodes), dtype=torch.long)
    node_position_tensor = torch.zeros((batch_size, max_nodes), dtype=torch.long)
    node_mask = torch.zeros((batch_size, max_nodes), dtype=torch.bool)

    target_tensor = torch.full((batch_size, max_target_len), fill_value=tgt_pad_id, dtype=torch.long)
    intent_tensor = torch.zeros(batch_size, dtype=torch.long)

    edge_src_parts: list[torch.Tensor] = []
    edge_dst_parts: list[torch.Tensor] = []
    edge_type_parts: list[torch.Tensor] = []

    sample_ids: list[str] = []
    references: list[str] = []
    reference_tokens: list[list[str]] = []

    node_offset = 0
    for batch_index, sample in enumerate(batch):
        node_count = len(sample.node_type_ids)
        sample_ids.append(sample.sample_id)
        references.append(sample.reference_summary)
        reference_tokens.append(sample.target_tokens)

        for node_index, token_ids in enumerate(sample.node_token_ids):
            node_token_tensor[batch_index, node_index, : len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)

        node_type_tensor[batch_index, :node_count] = torch.tensor(sample.node_type_ids, dtype=torch.long)
        node_position_tensor[batch_index, :node_count] = torch.tensor(sample.node_positions, dtype=torch.long)
        node_mask[batch_index, :node_count] = True
        target_tensor[batch_index, : len(sample.target_ids)] = torch.tensor(sample.target_ids, dtype=torch.long)
        intent_tensor[batch_index] = sample.intent_id

        if sample.edge_src:
            edge_src_parts.append(torch.tensor(sample.edge_src, dtype=torch.long) + node_offset)
            edge_dst_parts.append(torch.tensor(sample.edge_dst, dtype=torch.long) + node_offset)
            edge_type_parts.append(torch.tensor(sample.edge_type_ids, dtype=torch.long))
        node_offset += node_count

    flat_node_token_ids = node_token_tensor[node_mask]
    flat_node_type_ids = node_type_tensor[node_mask]
    flat_node_positions = node_position_tensor[node_mask]
    flat_graph_ids = torch.repeat_interleave(torch.arange(batch_size, dtype=torch.long), torch.tensor(graph_sizes))

    if edge_src_parts:
        edge_src = torch.cat(edge_src_parts, dim=0)
        edge_dst = torch.cat(edge_dst_parts, dim=0)
        edge_types = torch.cat(edge_type_parts, dim=0)
    else:
        edge_src = torch.empty(0, dtype=torch.long)
        edge_dst = torch.empty(0, dtype=torch.long)
        edge_types = torch.empty(0, dtype=torch.long)

    return {
        "sample_ids": sample_ids,
        "references": references,
        "reference_tokens": reference_tokens,
        "node_token_ids": flat_node_token_ids,
        "node_type_ids": flat_node_type_ids,
        "node_positions": flat_node_positions,
        "graph_ids": flat_graph_ids,
        "graph_sizes": torch.tensor(graph_sizes, dtype=torch.long),
        "edge_src": edge_src,
        "edge_dst": edge_dst,
        "edge_types": edge_types,
        "targets": target_tensor,
        "intent_ids": intent_tensor,
        "node_mask": node_mask,
    }
