"""Encode a raw Ref2VA JSONL manifest into training-ready parquet shards."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-frames", type=int, default=73)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--negative-prompt", default=None, help="Also prepare CFG caches; native default prompt is one space."
    )
    args = parser.parse_args()
    if args.shard_size < 1:
        parser.error("--shard-size must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be empty to avoid mixing caches")

    from veomni.models.diffusers.minimax_h3.minimax_h3_condition.configuration_minimax_h3_condition import (
        MiniMaxH3ConditionModelConfig,
    )
    from veomni.models.diffusers.minimax_h3.minimax_h3_condition.modeling_minimax_h3_condition import (
        MiniMaxH3ConditionModel,
    )
    from veomni.models.diffusers.minimax_h3.preprocessing import load_ref2va_record, prepare_ref2va_sample
    from veomni.trainer.dit_trainer import OfflineEmbeddingSaver

    with args.manifest.open() as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    if not records:
        parser.error("--manifest must contain at least one sample")
    condition = MiniMaxH3ConditionModel(MiniMaxH3ConditionModelConfig(base_model_path=args.model_path)).eval()
    shard_count = (len(records) + args.shard_size - 1) // args.shard_size
    saver = OfflineEmbeddingSaver(
        str(args.output),
        dataset_length=len(records),
        shard_num=shard_count,
        max_shard=shard_count,
    )
    for index, record in enumerate(records):
        decoded = load_ref2va_record(
            record,
            root=args.manifest.resolve().parent,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
        )
        sample = prepare_ref2va_sample(
            condition, **decoded, device=args.device, seed=args.seed + index, negative_prompt=args.negative_prompt
        )
        saver.save(sample)
    saver.save_last()


if __name__ == "__main__":
    main()
