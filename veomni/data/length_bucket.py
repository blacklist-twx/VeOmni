"""Stateful length grouping of global fixed-size batches before DP sharding."""

import hashlib
import itertools
import json
import math
from numbers import Integral

import torch
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler


class LengthBucketDistributedSampler(StatefulDistributedSampler):
    """Group similar lengths in shuffled windows, then stripe each batch over DP.

    Lengths must describe the dataset's untransformed index order. The sampler
    never reads samples, decodes media, or changes the number of samples per step.
    """

    def __init__(
        self,
        dataset,
        lengths,
        global_batch_size,
        *,
        window_batches=50,
        num_replicas=1,
        rank=0,
        shuffle=True,
        seed=0,
        drop_last=True,
    ):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed)
        if not isinstance(global_batch_size, int) or global_batch_size < 1 or global_batch_size % num_replicas:
            raise ValueError("global_batch_size must be positive and divisible by the DP size.")
        if not isinstance(window_batches, int) or window_batches < 1:
            raise ValueError("window_batches must be a positive integer.")
        self.lengths = list(lengths)
        if len(self.lengths) != len(dataset) or any(
            isinstance(n, bool) or not isinstance(n, Integral) or n <= 0 for n in self.lengths
        ):
            raise ValueError("Bucketing requires one positive integer length per dataset row.")
        self.lengths = [int(n) for n in self.lengths]
        self.global_batch_size = global_batch_size
        self.window_batches = window_batches
        self.drop_last = drop_last
        batches = len(dataset) // global_batch_size if drop_last else math.ceil(len(dataset) / global_batch_size)
        self.total_size = batches * global_batch_size
        self.num_samples = self.total_size // num_replicas
        contract = (self.lengths, global_batch_size, window_batches, num_replicas, rank, shuffle, seed, drop_last)
        self._fingerprint = hashlib.sha256(json.dumps(contract).encode()).hexdigest()

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = (
            torch.randperm(len(self.lengths), generator=generator).tolist()
            if self.shuffle
            else list(range(len(self.lengths)))
        )
        if self.total_size > len(indices):
            indices += (indices * math.ceil((self.total_size - len(indices)) / len(indices)))[
                : self.total_size - len(indices)
            ]
        else:
            indices = indices[: self.total_size]
        width = self.global_batch_size * self.window_batches
        batches = []
        for start in range(0, len(indices), width):
            window = sorted(indices[start : start + width], key=self.lengths.__getitem__)
            batches.extend(
                window[i : i + self.global_batch_size] for i in range(0, len(window), self.global_batch_size)
            )
        if self.shuffle:
            batches = [batches[i] for i in torch.randperm(len(batches), generator=generator).tolist()]
        local = (idx for batch in batches for idx in batch[self.rank :: self.num_replicas])
        self.yielded = self.next_yielded if self.next_yielded is not None else 0
        self.next_yielded = None
        for idx in itertools.islice(local, self.yielded, None):
            self.yielded += 1
            yield idx

    def set_epoch(self, epoch):
        if epoch != self.epoch:
            self.next_yielded = None
            self.yielded = 0
        super().set_epoch(epoch)

    def state_dict(self):
        return {
            "yielded": self.yielded if self.next_yielded is None else self.next_yielded,
            "epoch": self.epoch,
            "length_bucket_fingerprint": self._fingerprint,
        }

    def load_state_dict(self, state_dict):
        if state_dict.get("length_bucket_fingerprint") != self._fingerprint:
            raise ValueError("Length bucket metadata or sampler configuration changed since the checkpoint.")
        cursor = state_dict.get("yielded")
        if type(cursor) is not int or not 0 <= cursor <= self.num_samples:
            raise ValueError("Length bucket checkpoint cursor must be an integer within the epoch size.")
        super().load_state_dict(state_dict)
        self.epoch = state_dict["epoch"]
