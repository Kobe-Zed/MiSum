from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]


@dataclass
class Vocab:
    stoi: dict[str, int]
    itos: list[str]

    @classmethod
    def build(
        cls,
        token_sequences: Iterable[Sequence[str]],
        min_freq: int = 1,
        max_size: int | None = None,
    ) -> "Vocab":
        counter: Counter[str] = Counter()
        for sequence in token_sequences:
            counter.update(sequence)

        items = [
            token
            for token, freq in counter.most_common()
            if freq >= min_freq and token not in SPECIAL_TOKENS
        ]
        if max_size is not None:
            items = items[: max(0, max_size - len(SPECIAL_TOKENS))]

        itos = list(SPECIAL_TOKENS) + items
        stoi = {token: index for index, token in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.stoi[BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.stoi[EOS_TOKEN]

    def __len__(self) -> int:
        return len(self.itos)

    def encode(
        self,
        tokens: Sequence[str],
        add_bos: bool = False,
        add_eos: bool = False,
        max_length: int | None = None,
    ) -> List[int]:
        ids: List[int] = []
        if add_bos:
            ids.append(self.bos_id)
        ids.extend(self.stoi.get(token, self.unk_id) for token in tokens)
        if add_eos:
            ids.append(self.eos_id)
        if max_length is not None:
            ids = ids[:max_length]
            if add_eos and ids and ids[-1] != self.eos_id:
                ids[-1] = self.eos_id
        return ids

    def decode(self, ids: Sequence[int], stop_at_eos: bool = True) -> List[str]:
        tokens: List[str] = []
        for token_id in ids:
            if token_id == self.eos_id and stop_at_eos:
                break
            if token_id in (self.pad_id, self.bos_id):
                continue
            if 0 <= token_id < len(self.itos):
                tokens.append(self.itos[token_id])
            else:
                tokens.append(UNK_TOKEN)
        return tokens

    def save(self, path: str | Path) -> None:
        payload = {"itos": self.itos}
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        itos = payload["itos"]
        stoi = {token: index for index, token in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)
