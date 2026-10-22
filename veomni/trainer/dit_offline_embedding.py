# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DiT parquet cache persistence, coupled to the existing job checkpoint cursor."""

import hashlib
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset

from ..checkpoint import layout
from ..distributed.parallel_state import get_parallel_state
from ..utils.dist_utils import raise_if_any_rank_failed
from .callbacks.global_state_callback import GlobalStateCallback


class OfflineEmbeddingSaver:
    """Flush before the cursor checkpoint; reopen only at its committed shard boundary."""

    def __init__(
        self,
        save_path,
        dataset_length=0,
        shard_num=1,
        max_shard=1000,
        *,
        identity=None,
        max_buffer_bytes=256 * 1024 * 1024,
        max_rows_per_shard=1000,
    ):
        for name, value in (("max_buffer_bytes", max_buffer_bytes), ("max_rows_per_shard", max_rows_per_shard)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        self.max_buffer_bytes = max_buffer_bytes
        self.max_rows_per_shard = max_rows_per_shard
        self.buffer_bytes = 0
        state = get_parallel_state()
        self.dp_rank = state.dp_rank
        self.dp_size = state.dp_size
        self.save_path = Path(save_path).resolve()
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.dataset_length = dataset_length
        shard_num = max(1, min(shard_num, max_shard // self.dp_size))
        self.batch_len = max(1, (dataset_length + shard_num - 1) // shard_num)
        self.identity = {
            "config": identity,
            "path": str(self.save_path),
            "dp_size": self.dp_size,
            "dp_rank": self.dp_rank,
            "length": dataset_length,
            "batch_len": self.batch_len,
            "max_buffer_bytes": self.max_buffer_bytes,
            "max_rows_per_shard": self.max_rows_per_shard,
        }
        self.buffer = []
        self.shards = []
        self.rest_len = dataset_length
        self.index = 0
        self._resume_state = None
        self._opened = False

    @property
    def complete_path(self):
        return self.save_path / ".offline_embedding_complete"

    def _path(self, index):
        return self.save_path / f"rank_{self.dp_rank}_shard_{index}.parquet"

    @staticmethod
    def _digest(path):
        with open(path, "rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()

    def validate_state(self, state):
        if state["identity"] != self.identity:
            raise ValueError("DiT offline cache source, configuration, output path or parallel topology changed.")
        if self.complete_path.exists():
            raise FileExistsError("DiT offline cache is complete; use a new output directory.")
        rows = 0
        for index, shard in enumerate(state["shards"]):
            path = self._path(index)
            if not path.is_file() or self._digest(path) != shard["sha256"]:
                raise ValueError(f"Missing or corrupt committed DiT cache shard: {path}")
            rows += shard["rows"]
        if rows != self.dataset_length - state["rest_len"]:
            raise ValueError("DiT cache cursor and committed row count disagree.")

    def load_state_dict(self, state):
        if self._opened:
            raise RuntimeError("Load the DiT cache cursor before opening the writer.")
        self._resume_state = state

    def open(self):
        if self._opened:
            return
        if self._resume_state is None:
            if list(self.save_path.glob("*.parquet")) or self.complete_path.exists():
                raise FileExistsError(
                    "DiT cache already contains output; resume its checkpoint or use a new directory."
                )
        else:
            self.validate_state(self._resume_state)
            self.shards = list(self._resume_state["shards"])
            self.index = len(self.shards)
            self.rest_len = self._resume_state["rest_len"]
            # A crash after a flush but before the cursor commit leaves replayable shards.
            for path in self.save_path.glob(f"rank_{self.dp_rank}_shard_*.parquet"):
                index = int(path.stem.rsplit("_", 1)[1])
                if index >= self.index:
                    path.unlink()
        for path in self.save_path.glob(f".rank_{self.dp_rank}_shard_*.parquet.tmp"):
            path.unlink()
        self._opened = True

    @staticmethod
    def _cpu_recursive(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {k: OfflineEmbeddingSaver._cpu_recursive(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(OfflineEmbeddingSaver._cpu_recursive(v) for v in value)
        return value

    def save(self, item):
        if not self._opened:
            self.open()
        if self.rest_len == 0:
            return  # Ignore the sampler's trailing padding rows.
        encoded = {k: pickle.dumps(self._cpu_recursive(v)) for k, v in item.items()}
        row_bytes = sum(len(value) for value in encoded.values())
        # Flush the preceding rows before admitting one that would exceed the budget.
        # A single oversized row stays intact and is written as its own shard below.
        if self.buffer and self.buffer_bytes + row_bytes > self.max_buffer_bytes:
            self.save_last()
        self.buffer.append(encoded)
        self.buffer_bytes += row_bytes
        self.rest_len -= 1
        if (
            len(self.buffer) >= min(self.batch_len, self.max_rows_per_shard)
            or self.buffer_bytes >= self.max_buffer_bytes
        ):
            self.save_last()

    def save_last(self):
        if not self.buffer:
            return
        path = self._path(self.index)
        temporary = path.with_name(f".{path.name}.tmp")
        Dataset.from_list(self.buffer).to_parquet(str(temporary))
        # No partially written parquet is visible to the offline dataset reader.
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        digest = self._digest(temporary)
        os.replace(temporary, path)
        self.shards.append({"rows": len(self.buffer), "sha256": digest})
        self.buffer = []
        self.buffer_bytes = 0
        self.index += 1

    def state_dict(self):
        self.save_last()
        return {"identity": self.identity, "shards": list(self.shards), "rest_len": self.rest_len}

    def finalize(self, *, mark_complete=True):
        if self.rest_len != 0:
            raise RuntimeError(f"DiT offline cache ended with {self.rest_len} source samples still missing.")
        self.save_last()
        if mark_complete:
            self.complete_path.touch()


class DiTOfflineEmbeddingCallback(GlobalStateCallback):
    """Use the standard cursor/checkpoint manifest, with strict cache resume validation."""

    def state_dict(self, state):
        payload = None
        error = None
        try:
            payload = super().state_dict(state)
            payload["dit_encoder_state"] = self.trainer.model.extra_state()
            payload["python_rng_state"] = random.getstate()
            payload["numpy_rng_state"] = np.random.get_state()
        except Exception as exc:  # noqa: BLE001 - a failed flush must fail on every rank
            error = exc
        raise_if_any_rank_failed(error, "flushing the DiT offline cache")
        return payload

    def _read_current(self, load_path):
        manifest = layout.read_manifest(load_path)
        if manifest is None or manifest["world_size"] != self.trainer.args.train.world_size:
            raise ValueError("DiT cache resume requires a complete checkpoint with the same world size.")
        payload = super()._read_current(load_path)
        if payload is None or payload.get("dit_encoder_state") is None:
            raise ValueError("Checkpoint has no DiT offline embedding state.")
        if payload["global_step"] != manifest["global_step"]:
            raise ValueError("DiT cache checkpoint does not match its commit manifest.")
        writer = self.trainer.offline_cache_writer
        if writer is not None:
            if not payload.get("train_dataloader") or payload.get("offline_cache_writer") is None:
                raise ValueError("DiT cache resume requires both the dataloader cursor and cache writer state.")
            writer.validate_state(payload["offline_cache_writer"])
        return payload

    def load_global_state(self):
        payload = super().load_global_state()
        if payload is not None:
            self.trainer.model.load_extra_state(payload["dit_encoder_state"])
            random.setstate(payload["python_rng_state"])
            np.random.set_state(payload["numpy_rng_state"])
        return payload
