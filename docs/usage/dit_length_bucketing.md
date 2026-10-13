# DiT sample length bucketing

DiT length bucketing groups similar estimated sequence lengths into fixed-size
global batches before distributing samples across data-parallel ranks. It changes
sample order, not the training objective, microbatch size, or accumulation count.
It is disabled by default.

## Configuration

Add the following to an existing DiT training recipe:

```yaml
data:
  datasets_type: mapping
  length_bucket_column: sequence_length
  length_bucket_window_batches: 50
  dataloader:
    type: native
    shuffle: true
    in_order: true
    drop_last: true
```

The dataset must contain a positive integer `sequence_length` column, stored as
ordinary metadata rather than pickled tensors. Each value describes the
corresponding source row **after preprocessing**. For H3 prepared samples, use
`packed['seq_len']`, including text, reference, target audio and target video
rows. A duration alone does not capture resolution or reference-token costs.
Other models may use their own consistent token-count estimate.

A transform must produce exactly one training sample per source row and must not
randomly change its geometry. Keep lengths aligned with the dataset when filtering
or replacing media. The loader reads only this metadata column to build the
schedule; it does not decode every sample to estimate lengths.

## Ordering, tails and resume

The sampler shuffles source indices with `seed + epoch`, sorts bounded windows
of `global_batch_size * length_bucket_window_batches` indices by length, builds
global batches, and shuffles their order. Each batch is striped across DP ranks;
SP ranks continue to receive the same batch through the existing DiT broadcast.
With `shuffle: false`, sorting remains enabled but is deterministic across epochs.

`drop_last: true` discards the final incomplete global batch. With `false`, source
indices are repeated to fill that batch, so every rank has complete microbatches
and the same number of optimizer steps. Repeated tail samples contribute to loss
normally. No samples are discarded separately by length bucket.

StatefulDataLoader checkpoints preserve the sampler cursor and epoch, including
worker prefetch state. Restore with the same dataset, metadata, seed, DP topology,
batch size and window size. The sampler rejects a changed length/configuration
fingerprint. It cannot detect changed media when its length stays identical.
Streaming/iterable datasets, out-of-order worker delivery, dynamic token batching,
and offline embedding generation are not supported by this option.

## Implementation contract

- Dataset owners provide the metadata column. Framework-provided
  [`MappingDataset.get_column(name)`](../../veomni/data/dataset.py#L75) returns raw source-column values in index order
  without calling the transform; missing columns raise an error.
- [`DiTTrainer._build_dataloader()`](../../veomni/trainer/dit_trainer.py#L464) reads the requested column once and passes it to
  [`build_native_dataloader()`](../../veomni/data/data_loader.py#L70). Other trainers keep their existing defaults.
- [`LengthBucketDistributedSampler`](../../veomni/data/length_bucket.py#L13) owns the global ordering and DP assignment;
  [`DistributedDataloader.set_epoch()`](../../veomni/data/data_loader.py#L55) and its checkpoint machinery call the
  sampler's epoch/state methods. Model code does not participate in bucketing.

These paths are covered by `tests/data/test_length_bucket.py`: complete epoch
coverage, equal DP batch counts, tail behavior, changed-metadata rejection,
metadata-only access, fixed microbatches, and exact multi-worker resume.

Grouping can reduce padding for padded models and differences in work between DP
ranks. H3 already packs samples without batch padding, so its main prospective
benefit is rank workload balance. Synthetic padding and squared-length cost checks
are correctness/proxy tests, not a GPU throughput benchmark. Measure step-time
median/p95, input wait, memory and throughput on the intended dataset before
claiming a speedup; sorting can also affect storage read locality.
