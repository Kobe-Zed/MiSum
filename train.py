from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from misum_repro.data import (
    MiSumDataset,
    build_source_corpus,
    build_target_corpus,
    collate_graph_samples,
    load_raw_split,
)
from misum_repro.metrics import compute_generation_metrics
from misum_repro.model import MiSumReproductionModel
from misum_repro.utils import choose_device, load_yaml, save_json, save_yaml, set_seed
from misum_repro.vocab import Vocab


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def build_datasets(config: dict[str, Any]) -> tuple[MiSumDataset, MiSumDataset, MiSumDataset, Vocab, Vocab]:
    dataset_config = config["dataset"]
    vocab_config = config["vocab"]

    train_records = load_raw_split(
        dataset_root=dataset_config["root"],
        split_name="train",
        limit=dataset_config.get("max_train_samples"),
    )
    valid_records = load_raw_split(
        dataset_root=dataset_config["root"],
        split_name="valid",
        limit=dataset_config.get("max_valid_samples"),
    )
    test_records = load_raw_split(
        dataset_root=dataset_config["root"],
        split_name="test",
        limit=dataset_config.get("max_test_samples"),
    )

    src_vocab = Vocab.build(
        build_source_corpus(train_records),
        min_freq=vocab_config["src_min_freq"],
        max_size=vocab_config.get("src_max_size"),
    )
    tgt_vocab = Vocab.build(
        build_target_corpus(train_records),
        min_freq=vocab_config["tgt_min_freq"],
        max_size=vocab_config.get("tgt_max_size"),
    )

    dataset_kwargs = {
        "src_vocab": src_vocab,
        "tgt_vocab": tgt_vocab,
        "max_pseudo_nodes": dataset_config["max_pseudo_nodes"],
        "max_asm_blocks": dataset_config["max_asm_blocks"],
        "max_node_tokens": dataset_config["max_node_tokens"],
        "max_target_tokens": dataset_config["max_target_tokens"],
    }

    train_dataset = MiSumDataset(records=train_records, **dataset_kwargs)
    valid_dataset = MiSumDataset(records=valid_records, **dataset_kwargs)
    test_dataset = MiSumDataset(records=test_records, **dataset_kwargs)
    return train_dataset, valid_dataset, test_dataset, src_vocab, tgt_vocab


def evaluate(
    model: MiSumReproductionModel,
    data_loader: DataLoader,
    tgt_vocab: Vocab,
    device: torch.device,
    max_new_tokens: int,
    split_name: str,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_steps = 0
    all_predictions: list[list[str]] = []
    all_references: list[list[str]] = []
    prediction_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"eval:{split_name}", leave=False):
            batch = move_batch_to_device(batch, device)
            logits = model(batch)
            targets = batch["targets"][:, 1:]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=tgt_vocab.pad_id,
            )
            total_loss += float(loss.item())
            total_steps += 1

            generated_ids = model.generate(
                batch=batch,
                bos_id=tgt_vocab.bos_id,
                max_new_tokens=max_new_tokens,
            )
            generated_ids = generated_ids.detach().cpu().tolist()

            for sample_id, reference_tokens, generated in zip(
                batch["sample_ids"],
                batch["reference_tokens"],
                generated_ids,
            ):
                prediction_tokens = tgt_vocab.decode(generated)
                all_predictions.append(prediction_tokens)
                all_references.append(reference_tokens)
                prediction_rows.append(
                    {
                        "sample_id": sample_id,
                        "reference": " ".join(reference_tokens),
                        "prediction": " ".join(prediction_tokens),
                    }
                )

    metrics = compute_generation_metrics(all_references, all_predictions)
    metrics["loss"] = total_loss / max(1, total_steps)

    if output_dir is not None:
        predictions_path = output_dir / f"predictions_{split_name}.jsonl"
        with predictions_path.open("w", encoding="utf-8") as handle:
            for row in prediction_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return metrics


def create_data_loader(
    dataset: MiSumDataset,
    batch_size: int,
    shuffle: bool,
    collate_fn: Any,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )


def train(config_path: str) -> None:
    config = load_yaml(config_path)
    set_seed(config["seed"])
    device = choose_device(config["device"])
    train_config = config["train"]
    dataloader_config = config.get("dataloader", {})

    if device.type == "cuda" and train_config.get("allow_tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True
    if train_config.get("matmul_precision"):
        torch.set_float32_matmul_precision(train_config["matmul_precision"])

    train_dataset, valid_dataset, test_dataset, src_vocab, tgt_vocab = build_datasets(config)

    output_dir = Path("artifacts") / config["run_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(output_dir / "config.yaml", config)
    src_vocab.save(output_dir / "src_vocab.json")
    tgt_vocab.save(output_dir / "tgt_vocab.json")

    collate_fn = partial(collate_graph_samples, src_pad_id=src_vocab.pad_id, tgt_pad_id=tgt_vocab.pad_id)
    num_workers = dataloader_config.get("num_workers", 0)
    pin_memory = dataloader_config.get("pin_memory", device.type == "cuda")
    persistent_workers = bool(num_workers > 0 and dataloader_config.get("persistent_workers", True))
    train_loader = create_data_loader(
        train_dataset,
        batch_size=train_config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    valid_loader = create_data_loader(
        valid_dataset,
        batch_size=train_config.get("eval_batch_size", train_config["batch_size"]),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    test_loader = create_data_loader(
        test_dataset,
        batch_size=train_config.get("eval_batch_size", train_config["batch_size"]),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    model = MiSumReproductionModel(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        src_pad_id=src_vocab.pad_id,
        tgt_pad_id=tgt_vocab.pad_id,
        d_model=config["model"]["d_model"],
        num_heads=config["model"]["num_heads"],
        num_layers=config["model"]["num_layers"],
        ff_multiplier=config["model"]["ff_multiplier"],
        max_positions=config["model"]["max_positions"],
        dropout=config["model"]["dropout"],
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
    )
    use_amp = bool(device.type == "cuda" and train_config.get("use_amp", False))
    amp_dtype_name = train_config.get("amp_dtype", "float16")
    amp_dtype = getattr(torch, amp_dtype_name)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    gradient_accumulation_steps = max(1, int(train_config.get("gradient_accumulation_steps", 1)))

    best_valid_bleu = float("-inf")
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, train_config["epochs"] + 1):
        model.train()
        progress = tqdm(train_loader, desc=f"train:{epoch}")
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(progress, start=1):
            batch = move_batch_to_device(batch, device)
            autocast_context = (
                torch.autocast(device_type=device.type, dtype=amp_dtype)
                if use_amp
                else nullcontext()
            )
            with autocast_context:
                logits = model(batch)
                targets = batch["targets"][:, 1:]
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=tgt_vocab.pad_id,
                )

            scaled_loss = loss / gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()

            if step % gradient_accumulation_steps == 0 or step == len(train_loader):
                if train_config.get("grad_clip") is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), train_config["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss.item())
            progress.set_postfix(loss=f"{running_loss / step:.4f}")

        valid_metrics = evaluate(
            model=model,
            data_loader=valid_loader,
            tgt_vocab=tgt_vocab,
            device=device,
            max_new_tokens=config["decode"]["max_new_tokens"],
            split_name="valid",
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, len(train_loader)),
            "valid": valid_metrics,
        }
        history.append(epoch_record)

        if valid_metrics["bleu4"] > best_valid_bleu:
            best_valid_bleu = valid_metrics["bleu4"]
            epochs_without_improvement = 0
            torch.save({"model": model.state_dict()}, output_dir / "best.pt")
        else:
            epochs_without_improvement += 1

        save_json(output_dir / "metrics.json", {"history": history})

        if epochs_without_improvement >= train_config["early_stopping_patience"]:
            break

    checkpoint = torch.load(output_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])

    valid_metrics = evaluate(
        model=model,
        data_loader=valid_loader,
        tgt_vocab=tgt_vocab,
        device=device,
        max_new_tokens=config["decode"]["max_new_tokens"],
        split_name="valid",
        output_dir=output_dir,
    )
    test_metrics = evaluate(
        model=model,
        data_loader=test_loader,
        tgt_vocab=tgt_vocab,
        device=device,
        max_new_tokens=config["decode"]["max_new_tokens"],
        split_name="test",
        output_dir=output_dir,
    )

    payload = {
        "history": history,
        "best_valid": valid_metrics,
        "test": test_metrics,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "amp": use_amp,
        "dataset_sizes": {
            "train": len(train_dataset),
            "valid": len(valid_dataset),
            "test": len(test_dataset),
        },
    }
    save_json(output_dir / "metrics.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the MiSum reproduction model.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
