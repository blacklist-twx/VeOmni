"""Fixed DiT microbatches through the existing condition, model and saver paths."""

import copy
import pickle
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from datasets import Dataset
from torch import nn
from transformers.modeling_outputs import ModelOutput

import veomni.data.data_loader as loader_module
import veomni.models.model_runtime as runtime_module
import veomni.trainer.base as base_module
import veomni.trainer.dit_trainer as dit_module
from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.data.dataset import MappingDataset
from veomni.trainer.base import BaseTrainer
from veomni.trainer.dit_trainer import (
    DiTDataArguments,
    DiTDataCollator,
    DiTModelArguments,
    DiTModelRuntime,
    DiTTrainer,
    DiTTrainingArguments,
    OfflineEmbeddingSaver,
    VeOmniDiTArguments,
)


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _trainer(monkeypatch, task, micro_batch_size, dp_size=1, dp_rank=0, shuffle=True):
    monkeypatch.setenv("WORLD_SIZE", str(dp_size))
    state = SimpleNamespace(dp_size=dp_size, dp_rank=dp_rank, sp_size=1, sp_enabled=False)
    monkeypatch.setattr("veomni.distributed.parallel_state.get_parallel_state", lambda: state)
    monkeypatch.setattr(dit_module, "get_parallel_state", lambda: state)
    monkeypatch.setattr("veomni.trainer.dit_offline_embedding.get_parallel_state", lambda: state)
    monkeypatch.setattr(loader_module, "get_parallel_state", lambda: state)
    monkeypatch.setattr(runtime_module, "get_parallel_state_by_name", lambda name: state)
    monkeypatch.setattr(dit_module, "use_parallel_state", lambda state: nullcontext())
    args = VeOmniDiTArguments(
        model=DiTModelArguments(config_path="unused"),
        data=DiTDataArguments(train_path="unused", datasets_type="mapping"),
        train=DiTTrainingArguments(
            training_task=task,
            micro_batch_size=micro_batch_size,
            global_batch_size=4 * dp_size,
        ),
    )
    args.data.dataloader.num_workers = 0
    args.data.dataloader.prefetch_factor = None
    args.data.dataloader.pin_memory = False
    args.data.dataloader.shuffle = shuffle
    args._train_steps = 1
    trainer = DiTTrainer.__new__(DiTTrainer)
    trainer.base = SimpleNamespace(args=args, _setup=Mock(return_value=torch.device("cpu")), LOG_SAMPLE=False)
    trainer._setup()
    trainer.base._setup.assert_called_once_with(args)
    assert trainer.base.device == torch.device("cpu")
    return trainer


def _load(trainer, samples):
    trainer.base.train_dataset = [(sample,) for sample in samples]
    trainer._build_dataloader()
    return iter(trainer.base.train_dataloader)


@pytest.mark.parametrize("task", ["offline_training", "online_training"])
@pytest.mark.parametrize("micro_batch_size", [1, 2, 4])
@pytest.mark.parametrize("dp_size", [1, 2])
def test_setup_and_loader_honor_fixed_microbatches(monkeypatch, task, micro_batch_size, dp_size):
    trainer = _trainer(monkeypatch, task, micro_batch_size, dp_size)
    train = trainer.base.args.train
    assert train.micro_batch_size == micro_batch_size
    assert train.dyn_bsz is False
    assert train.dataloader_batch_size == 4
    assert train.gradient_accumulation_steps == 4 // micro_batch_size
    batches = next(_load(trainer, [{"value": torch.ones(i + 1)} for i in range(4 * dp_size)]))
    assert len(batches) == 4 // micro_batch_size
    assert all(len(batch["value"]) == micro_batch_size for batch in batches)
    assert all(isinstance(batch["value"], list) for batch in batches)


@pytest.mark.parametrize("task", ["offline_training", "online_training", "offline_embedding"])
@pytest.mark.parametrize("shuffle", [True, False])
def test_dataloader_shuffle_reaches_sampler(monkeypatch, task, shuffle):
    trainer = _trainer(monkeypatch, task, micro_batch_size=1, shuffle=shuffle)
    expected = shuffle and task != "offline_embedding"
    assert trainer.base.args.data.dataloader.shuffle is expected
    _load(trainer, [{"value": torch.ones(1)} for _ in range(4)])
    assert trainer.base.train_dataloader.sampler.shuffle is expected


@pytest.mark.parametrize("task", ["offline_training", "online_training", "offline_embedding"])
def test_offline_embedding_keeps_the_last_batch(monkeypatch, task):
    trainer = _trainer(monkeypatch, task, micro_batch_size=1)
    expected = task != "offline_embedding"
    assert trainer.base.args.data.dataloader.drop_last is expected
    assert not hasattr(trainer.base.args.data, "drop_last")
    _load(trainer, [{"value": torch.ones(1)} for _ in range(4)])
    assert trainer.base.train_dataloader.drop_last is expected


@pytest.mark.parametrize("shuffle", [True, False])
def test_dataloader_shuffle_reaches_base_trainer_builders(monkeypatch, shuffle):
    trainer = _trainer(monkeypatch, "offline_training", micro_batch_size=1, shuffle=shuffle)
    built = {}

    def build_dataset(**kwargs):
        built.update(kwargs)
        return list(range(4))

    monkeypatch.setattr(base_module, "build_dataset", build_dataset)
    base = SimpleNamespace(args=trainer.base.args, data_transform=None)
    BaseTrainer._build_dataset(base)
    assert built["shuffle"] is shuffle

    base.train_dataset = [({"value": torch.ones(1)},) for _ in range(4)]
    base.collate_fn = DiTDataCollator()
    BaseTrainer._build_dataloader(base)
    assert base.train_dataloader.sampler.shuffle is shuffle


class _Condition:
    def __init__(self):
        self.encoded = []
        self.processed = []

    def get_condition(self, *, value):
        assert not torch.is_grad_enabled()
        self.encoded.append(len(value))
        return {"value": [x + 1 for x in value]}

    def process_condition(self, *, value):
        assert not torch.is_grad_enabled()
        self.processed.append(len(value))
        return {"value": value}


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))

    def forward(self, *, value):
        losses = [(x * self.weight).square().mean() for x in value]
        return ModelOutput(loss={"mse": torch.stack(losses).mean()})


@pytest.mark.parametrize("task", ["offline_training", "online_training"])
@pytest.mark.parametrize("micro_batch_size", [1, 2, 4])
def test_train_step_matches_equal_sample_adamw_update(monkeypatch, task, micro_batch_size):
    trainer = _trainer(monkeypatch, task, micro_batch_size)
    model = _Model()
    reference = copy.deepcopy(model)
    condition = _Condition()
    runtime = DiTModelRuntime.__new__(DiTModelRuntime)
    runtime.model_name = "base"
    runtime.model = model
    runtime.condition_model = condition
    runtime.optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    runtime.lr_scheduler = torch.optim.lr_scheduler.StepLR(runtime.optimizer, step_size=1, gamma=0.5)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01)
    gradients = []

    def clip():
        gradients.append(model.weight.grad.clone())
        return model.weight.grad.norm()

    runtime.clip_grad_norm = clip
    trainer.base.model = runtime
    trainer.base.state = SimpleNamespace(global_step=0)
    trainer.base.model_fwd_context = nullcontext()
    trainer.base.model_bwd_context = nullcontext()
    for name in (
        "on_step_begin",
        "on_step_end",
        "sync_before_train_step",
        "_reset_async_activation_offload_if_enabled",
        "model_reshard",
        "_configure_hsdp_allreduce",
    ):
        setattr(trainer.base, name, Mock())
    values = [torch.arange(n, dtype=torch.float32) + i for i, n in enumerate((1, 3, 2, 5))]
    root_calls = []
    handle = model.register_forward_pre_hook(lambda module, args: root_calls.append(1))
    trainer.train_step(_load(trainer, [{"value": x} for x in values]))
    handle.remove()
    expected_values = [x + 1 for x in values] if task == "online_training" else values
    expected_loss = reference(value=expected_values).loss["mse"]
    expected_loss.backward()
    torch.testing.assert_close(gradients[0], reference.weight.grad)
    reference_optimizer.step()
    torch.testing.assert_close(model.weight, reference.weight)
    for key, value in reference_optimizer.state[reference.weight].items():
        torch.testing.assert_close(runtime.optimizer.state[model.weight][key], value)
    k = 4 // micro_batch_size
    assert len(root_calls) == k
    assert condition.processed == [micro_batch_size] * k
    assert condition.encoded == ([micro_batch_size] * k if task == "online_training" else [])
    assert trainer.base.state.global_step == 1
    assert runtime.lr_scheduler.last_epoch == 1
    assert runtime.optimizer.param_groups[0]["lr"] == 0.005
    assert model.weight.grad is None
    torch.testing.assert_close(torch.tensor(trainer.base.on_step_end.call_args.kwargs["loss"]), expected_loss.detach())


@pytest.mark.parametrize("micro_batch_size", [1, 2, 4])
@pytest.mark.parametrize("dp_size", [1, 2])
def test_embedding_keeps_one_microbatch_and_saves_each_sample(monkeypatch, tmp_path, micro_batch_size, dp_size):
    trainer = _trainer(monkeypatch, "offline_embedding", micro_batch_size, dp_size)
    train = trainer.base.args.train
    assert train.micro_batch_size == micro_batch_size
    assert train.global_batch_size == micro_batch_size * dp_size
    assert train.dataloader_batch_size == micro_batch_size
    assert train.gradient_accumulation_steps == 1
    condition = _Condition()
    trainer.base.model = SimpleNamespace(condition_model=condition)
    trainer.offline_embedding_saver = OfflineEmbeddingSaver(str(tmp_path), dataset_length=micro_batch_size)
    trainer.offline_embedding_saver.open()
    batches = next(_load(trainer, [{"value": torch.tensor([float(i)])} for i in range(micro_batch_size * dp_size)]))
    assert len(batches) == 1
    expected = [x + 1 for x in batches[0]["value"]]
    assert trainer.forward_backward_step(batches[0]) == (0.0, {})
    trainer.offline_embedding_saver.save_last()
    saved = Dataset.from_parquet(str(tmp_path / "rank_0_shard_0.parquet"))
    assert len(saved) == micro_batch_size
    for row, value in zip(saved, expected):
        torch.testing.assert_close(pickle.loads(row["value"]), value)
    assert condition.encoded == [micro_batch_size]
    assert condition.processed == []


@pytest.mark.parametrize("micro_batch_size", [1, 2, 4])
@pytest.mark.parametrize("dp_size", [1, 2])
def test_embedding_tail_padding_saves_original_samples_exactly_once(monkeypatch, tmp_path, micro_batch_size, dp_size):
    saved_ids = []
    for rank in range(dp_size):
        trainer = _trainer(monkeypatch, "offline_embedding", micro_batch_size, dp_size, rank)
        dataset = MappingDataset([{"value": torch.tensor([float(i)])} for i in range(5)], transform=lambda row: [row])
        trainer.base.train_dataset = dataset
        trainer.base._build_dataset = Mock()
        trainer.offline_embedding_save_dir = str(tmp_path / f"rank_{rank}")
        trainer.base.model = SimpleNamespace(condition_model=_Condition())
        trainer._build_dataset()
        trainer.offline_embedding_saver.open()
        trainer._build_dataloader()
        assert len(trainer.base.train_dataloader) == trainer.base.args.train_steps
        for micro_batches in trainer.base.train_dataloader:
            assert len(micro_batches) == 1
            trainer.forward_backward_step(micro_batches[0])
        trainer.offline_embedding_saver.save_last()
        rows = Dataset.from_parquet(str(tmp_path / f"rank_{rank}" / f"rank_{rank}_shard_0.parquet"))
        saved_ids.extend(int(pickle.loads(row["value"]).item()) - 1 for row in rows)
    assert sorted(saved_ids) == list(range(5))


def test_collator_aligns_missing_keys_with_none():
    single = DiTDataCollator()([{"a": 1, "b": 2}])
    assert dict(single) == {"a": [1], "b": [2]}
    mixed = DiTDataCollator()([{"a": 1, "x": "fl2va"}, {"a": 2, "y": "ref2va"}, {"a": 3}])
    assert dict(mixed) == {"a": [1, 2, 3], "x": ["fl2va", None, None], "y": [None, "ref2va", None]}


class _RandomCondition(_Condition):
    def __init__(self):
        super().__init__()
        self.generator = torch.Generator().manual_seed(123)

    def get_condition(self, *, value, augmentation):
        import random

        import numpy as np

        return {
            "value": value,
            "augmentation": augmentation,
            "noise": [
                torch.rand(3, generator=self.generator) + torch.rand(3) + random.random() + np.random.rand()
                for _ in value
            ],
        }


def _cache_transform(row):
    import random

    import numpy as np

    return [dict(row, augmentation=torch.rand(2) + random.random() + np.random.rand())]


def _cache_trainer(monkeypatch, tmp_path, *, rank=0, dp_size=1, load_path=None, num_workers=0):
    from veomni.trainer.dit_offline_embedding import DiTOfflineEmbeddingCallback

    trainer = _trainer(monkeypatch, "offline_embedding", micro_batch_size=2, dp_size=dp_size, dp_rank=rank)
    args = trainer.base.args
    args.data.dataloader.num_workers = num_workers
    args.data.dataloader.prefetch_factor = 2 if num_workers else None
    args.train.global_rank = rank
    args.train.checkpoint.save_path = str(tmp_path / "checkpoints")
    args.train.checkpoint.load_path = load_path
    args.train.checkpoint.save_steps = 2
    trainer._setup()
    trainer.base.train_dataset = MappingDataset(
        Dataset.from_dict({"value": list(range(13))}), transform=_cache_transform
    )
    trainer.base._build_dataset = Mock()
    trainer.offline_embedding_save_dir = str(tmp_path / "cache")
    condition = _RandomCondition()
    trainer.base.model = SimpleNamespace(
        condition_model=condition,
        parallel_state="base",
        extra_state=lambda: {"generator": condition.generator.get_state()},
        load_extra_state=lambda state: condition.generator.set_state(state["generator"]),
    )
    trainer._build_dataset()
    trainer._build_dataloader()
    trainer.base.offline_cache_writer = trainer.offline_embedding_saver
    trainer.base.saves_model_state = False
    trainer.base.environ_meter = Mock(state_dict=lambda: {}, load_state_dict=lambda _: None)
    trainer.base.state = SimpleNamespace(global_step=0, epoch=0)
    trainer.base.start_step = trainer.base.start_epoch = 0
    callback = DiTOfflineEmbeddingCallback(trainer.base)
    trainer.base.global_state_callback = callback
    trainer.base.on_train_begin = lambda: callback.on_train_begin(trainer.base.state)
    trainer.base.on_step_end = lambda **kwargs: callback.on_step_end(trainer.base.state)
    for name in ("on_step_begin", "sync_before_train_step", "_reset_async_activation_offload_if_enabled"):
        setattr(trainer.base, name, Mock())
    trainer.on_train_begin()
    from veomni.trainer.base import VeOmniIter

    trainer.base.data_iterator = VeOmniIter(trainer.base.train_dataloader)
    return trainer


def _cache_rows(path):
    rows = []
    for shard in sorted(path.glob("*.parquet")):
        rows.extend(
            {key: pickle.loads(value) for key, value in row.items()} for row in Dataset.from_parquet(str(shard))
        )
    return sorted(rows, key=lambda row: row["value"])


@pytest.mark.parametrize("crash", ["buffered", "flushed", "partial"])
def test_offline_cache_resume_matches_uninterrupted_trainer(monkeypatch, tmp_path, crash):
    import random

    import numpy as np

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    reference = _cache_trainer(monkeypatch, tmp_path / "reference")
    for _ in range(reference.base.args.train_steps):
        reference.train_step(reference.base.data_iterator)
    reference.offline_embedding_saver.finalize()
    expected = _cache_rows(tmp_path / "reference" / "cache")

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    interrupted = _cache_trainer(monkeypatch, tmp_path / "resumed")
    for _ in range(3):
        interrupted.train_step(interrupted.base.data_iterator)
    writer = interrupted.offline_embedding_saver
    assert len(writer.buffer) == 2  # checkpoint at step 2 covers four rows
    if crash == "flushed":
        writer.save_last()  # parquet exists, but no corresponding cursor checkpoint
    if crash == "partial":
        (writer.save_path / f".rank_0_shard_{writer.index}.parquet.tmp").write_bytes(b"interrupted parquet")
    resumed = _cache_trainer(
        monkeypatch,
        tmp_path / "resumed",
        load_path=str(tmp_path / "resumed/checkpoints/global_step_2"),
    )
    assert resumed.base.start_step == 2
    assert len(list(resumed.offline_embedding_saver.save_path.glob("*.parquet"))) == 1
    for _ in range(resumed.base.start_step, resumed.base.args.train_steps):
        resumed.train_step(resumed.base.data_iterator)
    resumed.offline_embedding_saver.finalize()
    actual = _cache_rows(tmp_path / "resumed" / "cache")
    assert len(actual) == len(expected) == 13
    for a, b in zip(actual, expected):
        assert a["value"] == b["value"]
        torch.testing.assert_close(a["noise"], b["noise"], rtol=0, atol=0)
        torch.testing.assert_close(a["augmentation"], b["augmentation"], rtol=0, atol=0)


@pytest.mark.parametrize(
    "damage", ["missing_cursor", "missing_manifest", "missing_shard", "corrupt_shard", "config", "complete"]
)
def test_offline_cache_rejects_unsafe_resume(monkeypatch, tmp_path, damage):
    from veomni.checkpoint import layout

    trainer = _cache_trainer(monkeypatch, tmp_path)
    for _ in range(2):
        trainer.train_step(trainer.base.data_iterator)
    load_path = layout.step_dir(str(tmp_path / "checkpoints"), 2)
    if damage == "missing_cursor":
        from pathlib import Path

        Path(layout.loader_path(load_path, 0)).unlink()
    elif damage == "missing_manifest":
        layout.remove_manifest(load_path)
    elif damage == "missing_shard":
        trainer.offline_embedding_saver._path(0).unlink()
    elif damage == "corrupt_shard":
        trainer.offline_embedding_saver._path(0).write_bytes(b"corrupt")
    elif damage == "config":
        payload_path = layout.extra_state_path(load_path, 0)
        payload = torch.load(payload_path, weights_only=False)
        payload["offline_cache_writer"]["identity"]["config"]["seed"] += 1
        torch.save(payload, payload_path)
    else:
        trainer.offline_embedding_saver.complete_path.touch()
    with pytest.raises((ValueError, FileExistsError), match="cache|checkpoint|cursor|state|source|shard"):
        _cache_trainer(monkeypatch, tmp_path, load_path=load_path)


def test_offline_cache_refuses_fresh_overwrite(monkeypatch, tmp_path):
    trainer = _cache_trainer(monkeypatch, tmp_path)
    trainer.train_step(trainer.base.data_iterator)
    trainer.offline_embedding_saver.save_last()
    with pytest.raises(FileExistsError, match="already contains"):
        _cache_trainer(monkeypatch, tmp_path)


def test_offline_saver_supports_direct_save_api(monkeypatch, tmp_path):
    _trainer(monkeypatch, "offline_embedding", 1)
    saver = OfflineEmbeddingSaver(str(tmp_path), dataset_length=1)
    saver.save({"value": 9})
    saver.save_last()
    assert _cache_rows(tmp_path) == [{"value": 9}]


def test_offline_cache_resume_rejects_worker_random_transforms(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="num_workers=0"):
        _cache_trainer(monkeypatch, tmp_path, num_workers=1)
