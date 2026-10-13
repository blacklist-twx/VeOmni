"""Length grouping coverage, distributed alignment and exact loader replay."""

import copy
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.data.data_loader import DistributedDataloader, build_native_dataloader
from veomni.data.dataset import MappingDataset
from veomni.data.length_bucket import LengthBucketDistributedSampler


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def sampler(lengths, **kwargs):
    return LengthBucketDistributedSampler(range(len(lengths)), lengths, 8, **kwargs)


@pytest.mark.parametrize("size", [0, 3, 8, 19, 80])
@pytest.mark.parametrize("drop", [True, False])
def test_global_tail_and_dp_coverage(size, drop):
    lengths = list(range(1, size + 1))
    streams = [list(sampler(lengths, num_replicas=2, rank=r, drop_last=drop)) for r in range(2)]
    assert len(streams[0]) == len(streams[1])
    assert len(streams[0]) % 4 == 0
    ids = streams[0] + streams[1]
    if drop:
        assert len(ids) == size // 8 * 8
        assert len(set(ids)) == len(ids)
    else:
        assert set(ids) == set(range(size))
        assert len(ids) == (size + 7) // 8 * 8


def test_epoch_shuffle_and_rank_resume():
    lengths = [i % 13 + 1 for i in range(80)]
    first = sampler(lengths, num_replicas=2, rank=1)
    epoch0 = list(first)
    first.set_epoch(3)
    it = iter(first)
    prefix = [next(it) for _ in range(7)]
    state = first.state_dict()
    suffix = list(it)
    resumed = sampler(lengths, num_replicas=2, rank=1)
    resumed.load_state_dict(state)
    assert list(resumed) == suffix
    assert prefix + suffix != epoch0
    assert prefix + suffix == list(first)
    changed = sampler(lengths[:-1] + [99], num_replicas=2, rank=1)
    with pytest.raises(ValueError, match="changed"):
        changed.load_state_dict(state)


@pytest.mark.parametrize("workers", [0, 2])
def test_stateful_loader_resume_with_prefetch(workers):
    lengths = [i % 17 + 1 for i in range(80)]

    def loader():
        options = {"multiprocessing_context": "spawn", "prefetch_factor": 2} if workers else {}
        return DistributedDataloader(
            list(range(80)),
            sampler=sampler(lengths),
            batch_size=8,
            num_workers=workers,
            **options,
        )

    original = loader()
    original.set_epoch(2)
    it = iter(original)
    next(it)
    next(it)
    state = copy.deepcopy(original.state_dict())
    expected = list(it)
    resumed = loader()
    resumed.load_state_dict(state)
    actual = list(resumed)
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_grouping_reduces_padding_and_rank_cost_proxy():
    lengths = [16, 1024, 32, 512, 64, 256, 128, 2048] * 16
    grouped = list(sampler(lengths, shuffle=False))

    def costs(order):
        padding, critical = 0, 0
        for start in range(0, len(order), 8):
            batch = [lengths[i] for i in order[start : start + 8]]
            padding += max(batch) * 8 - sum(batch)
            critical += max(sum(n * n for n in batch[r::2]) for r in range(2))
        return padding, critical

    before = costs(list(range(len(lengths))))
    after = costs(grouped)
    assert after[0] < before[0]
    assert after[1] < before[1]


@pytest.mark.parametrize("bad", [[1, 0], [1, -1], [1, 2.5], [1, True], [1]])
def test_bad_metadata_is_rejected(bad):
    with pytest.raises(ValueError, match="positive integer"):
        LengthBucketDistributedSampler(range(2), bad, 2)


def test_metadata_read_does_not_run_transform():
    def forbidden(_):
        raise AssertionError("Metadata lookup must not decode a sample")

    data = MappingDataset(Dataset.from_dict({"length": [1, 9, 3]}), transform=forbidden)
    assert data.get_column("length") == [1, 9, 3]
    with pytest.raises(ValueError, match="Missing"):
        data.get_column("absent")


def test_native_fixed_microbatches_use_grouped_sampler(monkeypatch):
    monkeypatch.setattr(
        "veomni.data.data_loader.get_parallel_state", lambda: SimpleNamespace(dp_size=2, dp_rank=1, sp_size=1)
    )
    lengths = [2, 32, 4, 16, 2, 32, 4, 16] * 2
    dataset = [({"id": i},) for i in range(len(lengths))]
    loader = build_native_dataloader(
        dataset,
        micro_batch_size=2,
        global_batch_size=8,
        dataloader_batch_size=4,
        max_seq_len=32,
        train_steps=2,
        dyn_bsz=False,
        num_workers=0,
        prefetch_factor=None,
        pin_memory=False,
        collate_fn=lambda samples: [s["id"] for s in samples],
        sample_lengths=lengths,
        shuffle=False,
    )
    batches = list(loader)
    expected = list(sampler(lengths, num_replicas=2, rank=1, shuffle=False))
    assert batches == [[expected[i : i + 2], expected[i + 2 : i + 4]] for i in range(0, len(expected), 4)]


def test_offline_cache_metadata_is_not_unpickled(monkeypatch):
    import pickle
    from functools import partial

    from veomni.data.multimodal.dit.data_transform import process_dit_offline_example

    monkeypatch.setattr(
        "veomni.data.data_loader.get_parallel_state", lambda: SimpleNamespace(dp_size=1, dp_rank=0, sp_size=1)
    )
    source = Dataset.from_dict(
        {"id": [pickle.dumps(i) for i in range(8)], "sequence_length": [8, 1, 7, 2, 6, 3, 5, 4]}
    )
    dataset = MappingDataset(
        source, transform=partial(process_dit_offline_example, length_bucket_column="sequence_length")
    )
    loader = build_native_dataloader(
        dataset,
        micro_batch_size=2,
        global_batch_size=4,
        dataloader_batch_size=4,
        max_seq_len=8,
        train_steps=2,
        dyn_bsz=False,
        num_workers=0,
        prefetch_factor=None,
        pin_memory=False,
        shuffle=False,
        collate_fn=lambda samples: [s["id"] for s in samples],
        sample_lengths=dataset.get_column("sequence_length"),
    )
    assert list(loader) == [[[1, 3], [5, 7]], [[6, 4], [2, 0]]]
    assert process_dit_offline_example({"id": pickle.dumps(2)}) == [{"id": 2}]


@pytest.mark.parametrize("workers", [0, 2])
def test_resume_at_epoch_boundary_starts_complete_next_epoch(workers):
    lengths = list(range(1, 81))

    def loader():
        options = {"multiprocessing_context": "spawn", "prefetch_factor": 2} if workers else {}
        return DistributedDataloader(
            list(range(80)), sampler=sampler(lengths), batch_size=8, num_workers=workers, **options
        )

    original = loader()
    iterator = iter(original)
    for _ in range(10):
        next(iterator)
    state = copy.deepcopy(original.state_dict())
    resumed = loader()
    resumed.load_state_dict(state)
    iter(resumed)  # GlobalStateCallback materializes the restored iterator before set_epoch.
    resumed.set_epoch(1)
    actual = list(resumed)
    fresh = loader()
    fresh.set_epoch(1)
    expected = list(fresh)
    assert len(actual) == len(expected) == 10
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_resaving_loaded_cursor_without_consuming_preserves_position():
    lengths = list(range(1, 81))

    def loader():
        return DistributedDataloader(list(range(80)), sampler=sampler(lengths), batch_size=8, num_workers=0)

    original = loader()
    it = iter(original)
    next(it)
    next(it)
    restored = loader()
    restored.load_state_dict(copy.deepcopy(original.state_dict()))
    replay = loader()
    replay.load_state_dict(copy.deepcopy(restored.state_dict()))
    actual, expected = list(replay), list(it)
    assert len(actual) == len(expected) == 8
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("cursor", [-1, 11, 1.5, True])
def test_invalid_resume_cursor_is_rejected(cursor):
    original = sampler(list(range(1, 81)), num_replicas=8)
    state = original.state_dict()
    state["yielded"] = cursor
    with pytest.raises(ValueError, match="cursor"):
        original.load_state_dict(state)
