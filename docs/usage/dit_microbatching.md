# DiT Fixed Microbatches

`DiTTrainer` honors `train.micro_batch_size` instead of overriding it to one.
Existing recipes with `micro_batch_size: 1` are unchanged. A larger microbatch
does not automatically enable packing or guarantee a speedup: each model decides
whether to loop over samples, batch or pack them, or reject unsupported inputs.

## Training batch sizes

DiT keeps `train.dyn_bsz = false`. With microbatch size `M`, data-parallel size
`D`, and `K` gradient-accumulation microbatches per optimizer step:

```text
G = M × D × K

dataloader_batch_size = G / D = M × K
```

For example, this overlay uses `K = 2` when `D = 4`, provided the chosen model
supports two samples per microbatch:

```yaml
train:
  training_task: offline_training
  dyn_bsz: false
  micro_batch_size: 2
  global_batch_size: 16
```

`MakeMicroBatchCollator` splits each dataloader batch into `K` microbatches.
`DiTDataCollator` keeps each one as a `dict[str, list]`, with `M` entries per
column. It does not concatenate tokens or slice inputs for sequence parallelism.

The existing training path is unchanged:

```text
get_condition? → process_condition(**batch) → model(**batch) → outputs.loss / K
```

`online_training` runs `get_condition` first; `offline_training` reads cached
conditions. Both use the same condition-processing and model-forward interfaces.
Each scalar in `outputs.loss` must be a **sample mean** within the microbatch.
Models must average per-sample losses, not all packed tokens, so longer samples
do not receive more weight. The trainer then divides by `K` for accumulation.

## Offline embedding

`offline_embedding` runs only the condition model and writes one parquet row per
sample; it never packs or invokes the DiT. It also honors `M`, but overrides the
training global batch to `G = M × D`, with `dataloader_batch_size = M` and `K = 1`.
There is no optimizer, and keeping one microbatch per step limits the amount of
video data serialized by the sequence-parallel broadcast. Choose `M` to fit the
encoder and host-memory budget. Embedding disables sampler shuffling and pads
only an index view of the source dataset; the saver discards trailing repeats
so each original sample is written once.

### Bounded cache buffering

Embedding flushes a shard when its serialized payload reaches 256 MiB or it
contains 1,000 rows, even when checkpointing is disabled. Configure the limits:

```yaml
data:
  offline_embedding_max_buffer_mb: 256
  offline_embedding_max_rows_per_shard: 1000
```

Both values must be positive integers. The byte budget counts pickled column
values, not process RSS or compressed parquet size; the current sample, encoder
outputs and parquet conversion also use memory. A row larger than the budget is
written alone. Explicit `OfflineEmbeddingSaver(shard_num=...)` splitting still
applies; these limits can produce additional, smaller shards. Resume with the
same limits as the checkpoint.

### Resuming embedding

Use the existing checkpoint options to save the cache position together with
its dataloader cursor. Each checkpoint flushes buffered rows to atomic parquet
shards before committing the cursor; it does not save model weights.

```yaml
data:
  offline_embedding_save_dir: /shared/cache/my_dataset
  dataloader:
    num_workers: 0
    in_order: true
    use_background_prefetcher: false
train:
  training_task: offline_embedding
  checkpoint:
    output_dir: /shared/runs/my_embedding
    save_steps: 100
    # On restart, set the complete step directory returned by this run:
    # load_path: /shared/runs/my_embedding/checkpoints/global_step_100
```

Resume with the same source dataset, condition-model configuration, batch size,
worker configuration and parallel topology. Resumable runs require the native
dataloader with `num_workers: 0`, `in_order: true` and
`use_background_prefetcher: false`, so random preprocessing runs in the
checkpointed process. Worker random streams are not restored by the current
dataloader cursor. The dataset fingerprint and resolved
configuration are checked. Keep referenced media and model weights immutable;
their contents are not rehashed. The output directory and checkpoint directory
must be on shared storage with atomic file replacement, and only one job may
write to them at a time.

Restart restores the dataloader cursor and encoder RNG, verifies committed
shards, and removes only shards written after that checkpoint. Those uncommitted
samples are encoded again. Missing cursor files, incomplete checkpoints, changed
configuration and missing or corrupt committed shards fail explicitly. A fresh
run refuses existing parquet output; a completed cache cannot be resumed.
Without `save_steps`, embedding keeps its existing one-pass behavior and no
restart checkpoint is created. Use only the final cache as training input.

## Model ownership

Packing, valid-token selection, multimodal layouts, attention boundaries,
positions and output reconstruction remain inside modeling. SP/CP slicing also
belongs there, after patchification where applicable, not in the DiT collator.
A model that cannot handle a multi-sample or parallel configuration must reject
it itself. The trainer adds no packing switch, alternate sample/output protocol,
or model-capability envelope.

This is fixed-sample batching, independent of the token-budget scheduling used
by dynamic batching. A larger microbatch does not imply packing for every model.
MiniMax H3 implements model-owned packing; see
[its usage guide](../examples/minimax_h3.md) for supported configurations.
