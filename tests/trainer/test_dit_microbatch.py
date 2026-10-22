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
import veomni.utils.helper as helper_module
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


def _meter(
    monkeypatch,
    model_type="MiniMaxH3DiTModel",
    dp_size=1,
    sp_size=1,
    remote_unknown=0,
    condition_model_type="condition",
    **config,
):
    dp_group = object()
    monkeypatch.setattr(helper_module.dist, "get_world_size", lambda: dp_size * sp_size)
    monkeypatch.setattr(helper_module, "compute_device_memory_metrics", lambda: {})
    counter = Mock()
    counter.estimate_flops.return_value = (1.0, 2.0)
    monkeypatch.setattr(helper_module, "VeomniFlopsCounter", lambda config: counter)

    def reduce(values, *, op, group):
        assert op == "sum"
        assert group is dp_group
        reduced = [value * dp_size for value in values]
        if remote_unknown:
            reduced[-1] += remote_unknown
        return reduced

    monkeypatch.setattr(helper_module, "all_reduce", reduce)
    meter = helper_module.EnvironMeter(
        config=SimpleNamespace(model_type=model_type, condition_model_type=condition_model_type, **config),
        global_batch_size=4 * dp_size,
        empty_cache_steps=0,
        parallel_state=SimpleNamespace(dp_group=dp_group),
    )
    return meter


def _packed_sample(video_len=9, audio_len=3):
    # The allocation includes padding, which is not a valid multimodal token.
    return {
        "packed": {
            "text_pos": torch.arange(2),
            "img_pos": torch.arange(video_len) + 2,
            "audio_pos": torch.arange(audio_len) + video_len + 2,
            "seq_len": 128,
        }
    }


@pytest.mark.parametrize("micro_batch_size", [1, 2, 4])
@pytest.mark.parametrize("dp_size,sp_size", [(1, 1), (2, 1), (2, 2)])
def test_dit_meter_counts_samples_independently_of_tokens(monkeypatch, micro_batch_size, dp_size, sp_size):
    meter = _meter(monkeypatch, dp_size=dp_size, sp_size=sp_size)
    samples = [_packed_sample(video_len=i + 1) for i in range(4)]
    for start in range(0, 4, micro_batch_size):
        meter.add(DiTDataCollator()(samples[start : start + micro_batch_size]))
    metrics = meter.step(delta_time=2.0, global_step=1)
    expected_tokens = sum(2 + i + 1 + 3 for i in range(4)) * dp_size
    assert metrics["consumed_chunk_num"] == 4 * dp_size
    assert metrics["samples_per_second"] == 2 * dp_size
    assert metrics["tokens_per_second(M)"] == expected_tokens / 2e6
    assert metrics["training/avg_sample_seq_len"] == expected_tokens / (4 * dp_size)
    assert metrics["token_count_available"] == 1


@pytest.mark.parametrize("shape", [(8, 4, 6, 8), (1, 8, 4, 6, 8), (2, 8, 4, 6, 8)])
@pytest.mark.parametrize("patch_size", [(1, 2, 2), (2, 3, 4)])
def test_dit_meter_wan_uses_configured_patch_grid(monkeypatch, shape, patch_size):
    meter = _meter(monkeypatch, model_type="WanTransformer3DModel", patch_size=patch_size)
    meter.add(DiTDataCollator()([{"latents": torch.zeros(shape), "context": torch.ones(1, 5, 8)}]))
    metrics = meter.step(delta_time=1.0, global_step=1)
    batch = shape[0] if len(shape) == 5 else 1
    expected = batch * (4 // patch_size[0]) * (6 // patch_size[1]) * (8 // patch_size[2])
    assert metrics["consumed_chunk_num"] == 1  # Counts dataset rows, independent of a tensor's axes.
    assert metrics["tokens_per_second(M)"] == expected / 1e6
    assert metrics["consume_tokens(M)"] == expected / 1e6


@pytest.mark.parametrize(
    "model_type,sample,config",
    [
        ("WanTransformer3DModel", {"videos": ["raw-video"], "inputs": "caption"}, {"patch_size": (1, 2, 2)}),
        ("MiniMaxH3DiTModel", {"input_latents": torch.zeros(1, 24, 2, 4, 4)}, {}),
        ("MiniMaxH3DiTModel", {"packed": {"img_pos": torch.arange(9)}}, {}),
        ("UnknownDiT", {"latents": torch.zeros(1, 8, 2, 4, 4)}, {"patch_size": (1, 2, 2)}),
        ("WanTransformer3DModel", {"latents": torch.zeros(1, 9, 8)}, {"patch_size": (1, 2, 2)}),
        ("WanTransformer3DModel", {"latents": torch.zeros(1, 8, 2, 3, 3)}, {"patch_size": (1, 2, 2)}),
        ("WanTransformer3DModel", {"latents": torch.zeros(1, 8, 2, 4, 4)}, {}),
    ],
)
def test_dit_meter_unknown_layout_counts_samples_without_inventing_tokens(monkeypatch, model_type, sample, config):
    meter = _meter(monkeypatch, model_type=model_type, dp_size=2, sp_size=2, **config)
    meter.add(DiTDataCollator()([sample, sample]))
    metrics = meter.step(delta_time=0.5, global_step=1)
    assert metrics == {
        "consumed_chunk_num": 4,
        "samples_per_second": 8,
        "token_count_available": 0,
        "cumulative_token_count_available": 0,
    }


@pytest.mark.parametrize("unknown_location", ["same_microbatch", "next_microbatch", "other_dp_rank"])
def test_dit_meter_never_reports_partial_tokens(monkeypatch, unknown_location):
    remote = unknown_location == "other_dp_rank"
    meter = _meter(monkeypatch, dp_size=2 if remote else 1, sp_size=2, remote_unknown=int(remote))
    known, unknown = _packed_sample(), {"inputs": "raw caption"}
    if remote:
        meter.add(DiTDataCollator()([known]))
    elif unknown_location == "same_microbatch":
        meter.add(DiTDataCollator()([known, unknown]))
    else:
        meter.add(DiTDataCollator()([known]))
        meter.add(DiTDataCollator()([unknown]))
    metrics = meter.step(delta_time=1.0, global_step=1)
    assert metrics == {
        "consumed_chunk_num": 2,
        "samples_per_second": 2,
        "token_count_available": 0,
        "cumulative_token_count_available": 0,
    }


@pytest.mark.parametrize("first_known", [True, False])
@pytest.mark.parametrize("resume", [True, False])
def test_dit_meter_resets_step_counts_and_preserves_cumulative_completeness(monkeypatch, first_known, resume):
    meter = _meter(monkeypatch, dp_size=2, sp_size=2)
    first = _packed_sample(audio_len=0) if first_known else {"inputs": "raw caption"}
    meter.add(DiTDataCollator()([first, first]))
    meter.step(delta_time=1.0, global_step=1)
    if resume:
        state = meter.state_dict()
        meter = _meter(monkeypatch, dp_size=2, sp_size=2)
        meter.load_state_dict(state)
    meter.add(DiTDataCollator()([_packed_sample()]))
    metrics = meter.step(delta_time=2.0, global_step=2)
    assert metrics["consumed_chunk_num"] == 6
    assert metrics["samples_per_second"] == 1
    assert metrics["tokens_per_second(M)"] == 14 / 1e6
    assert metrics["token_count_available"] == 1
    assert metrics["cumulative_token_count_available"] == int(first_known)
    if first_known:
        assert metrics["consume_tokens(M)"] == (11 * 4 + 14 * 2) / 1e6
    else:
        assert "consume_tokens(M)" not in metrics
        assert "consume_tokens(B)" not in metrics


def test_dit_meter_old_checkpoint_does_not_claim_complete_token_totals(monkeypatch):
    meter = _meter(monkeypatch)
    meter.load_state_dict({"consume_tokens": 100, "consume_chunks": 7})
    meter.add(DiTDataCollator()([_packed_sample()]))
    metrics = meter.step(delta_time=1.0, global_step=2)
    assert metrics["consumed_chunk_num"] == 8
    assert metrics["token_count_available"] == 1
    assert metrics["cumulative_token_count_available"] == 0
    assert "consume_tokens(M)" not in metrics


def test_dit_meter_changes_preserve_text_metrics_and_checkpoint_schema(monkeypatch):
    meter = _meter(monkeypatch, model_type="qwen3", condition_model_type=None, dp_size=2, sp_size=2)
    meter.add({"cu_seq_lens_q": torch.tensor([0, 3, 8, 10]), "tail_padding_length": 2})
    meter.add({"attention_mask": torch.tensor([[1, 1, 0]])})
    metrics = meter.step(delta_time=2.0, global_step=1)
    assert metrics["consumed_chunk_num"] == 6
    assert metrics["tokens_per_second(M)"] == 10 / 1e6
    assert metrics["consume_tokens(M)"] == 20 / 1e6
    assert metrics["training/avg_effective_len"] == 2.5
    assert metrics["training/avg_sample_seq_len"] == 20 / 6
    assert metrics["flops_achieved(T)"] == 2.0
    assert metrics["flops_promised(T)"] == 8.0
    assert metrics["mfu"] == 0.25
    assert "samples_per_second" not in metrics
    assert "token_count_available" not in metrics
    assert meter.state_dict() == {"consume_tokens": 20, "consume_chunks": 6}
    restored = _meter(monkeypatch, model_type="qwen3", condition_model_type=None, dp_size=2, sp_size=2)
    restored.load_state_dict(meter.state_dict())
    restored.add({"attention_mask": torch.ones(1, 3)})
    assert restored.step(delta_time=1.0, global_step=2)["consume_tokens(M)"] == 26 / 1e6


def test_legacy_dit_meter_keeps_explicit_sample_count(monkeypatch):
    from veomni.utils.dit_utils import EnvironMeter

    meter = _meter(monkeypatch, model_type="WanTransformer3DModel", patch_size=(1, 2, 2))
    monkeypatch.setattr(helper_module, "get_parallel_state", lambda: meter.parallel_state)
    legacy = EnvironMeter(config=meter.config, global_batch_size=1, empty_cache_steps=0)
    legacy.add({"latents": torch.zeros(1, 8, 2, 4, 4)}, model_type="wan")
    metrics = legacy.step(delta_time=1.0, global_step=1)
    assert metrics["consumed_chunk_num"] == 1
    assert metrics["tokens_per_second(M)"] == 8 / 1e6


def _trainer(monkeypatch, task, micro_batch_size, dp_size=1, dp_rank=0, shuffle=True):
    monkeypatch.setenv("WORLD_SIZE", str(dp_size))
    state = SimpleNamespace(dp_size=dp_size, dp_rank=dp_rank, sp_size=1, sp_enabled=False)
    monkeypatch.setattr("veomni.distributed.parallel_state.get_parallel_state", lambda: state)
    monkeypatch.setattr(dit_module, "get_parallel_state", lambda: state)
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
        trainer.offline_embedding_save_dir = str(tmp_path)
        trainer.base.model = SimpleNamespace(condition_model=_Condition())
        trainer._build_dataset()
        trainer._build_dataloader()
        assert len(trainer.base.train_dataloader) == trainer.base.args.train_steps
        for micro_batches in trainer.base.train_dataloader:
            assert len(micro_batches) == 1
            trainer.forward_backward_step(micro_batches[0])
        trainer.offline_embedding_saver.save_last()
        rows = Dataset.from_parquet(str(tmp_path / f"rank_{rank}_shard_0.parquet"))
        saved_ids.extend(int(pickle.loads(row["value"]).item()) - 1 for row in rows)
    assert sorted(saved_ids) == list(range(5))


def test_collator_aligns_missing_keys_with_none():
    single = DiTDataCollator()([{"a": 1, "b": 2}])
    assert dict(single) == {"a": [1], "b": [2]}
    mixed = DiTDataCollator()([{"a": 1, "x": "fl2va"}, {"a": 2, "y": "ref2va"}, {"a": 3}])
    assert dict(mixed) == {"a": [1, 2, 3], "x": ["fl2va", None, None], "y": [None, "ref2va", None]}
