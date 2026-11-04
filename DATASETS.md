# Dataset format

Run commands from the source repository root. Benchmark data and trained
checkpoints are obtained separately.

## Prepare four-column split files

Create a source directory containing `train.txt`, `valid.txt`, and `test.txt`
(or `.del` files). Each row is whitespace-separated:

```text
subject relation object timestamp
```

For example, a row can use string entity labels and an ISO date:

```text
entity_a relation_1 entity_b 2014-01-01
```

Numeric entity IDs and numeric timestamps are also supported. Timestamps must
be numeric or ISO date/datetime labels so that their chronological order is
defined. An existing `time_ids.del` file can map input time IDs to date labels.

```bash
python tools/prepare_dataset.py --source raw/GDELT --output data/gdelt
python tools/prepare_dataset.py --source raw/ICEWS18 --output data/icews18
python tools/prepare_dataset.py --source raw/ICEWS05-15 --output data/icews05-15-chrono
```

Output layout:

```text
data/gdelt/
  dataset.yaml
  entity_ids.del
  relation_ids.del
  time_ids.del
  train.del
  valid.del
  test.del
```

The YAML metadata tells the loader which files to read. Split membership,
row order, and duplicate events are preserved. Entity/relation ID maps are
retained when supplied with the input. Otherwise IDs are assigned consistently
over the combined vocabulary. Timestamps are converted to chronological ordinal
IDs; this is not an elapsed-time conversion. For a matched existing experiment,
use its processed files and ID maps rather than creating new mappings.

If a processed ICEWS05-15 dataset is already at `data/icews05-15/`, the following
shortcut creates a chronological sibling directory:

```bash
python data/remap_icews05_15_time.py
```

The utility creates a new output directory and preserves an existing one.

## Benchmark configurations

GDELT and ICEWS05-15 use the overlapping-date benchmark versions declared in
their YAML configurations. ICEWS18 uses the chronological benchmark version.
The repository supplies those training configurations; it does not substitute
another dataset sharing a similar name.

The loader expects the candidate entity vocabulary to be represented by
`entity_ids.del`. Include entities declared by the dataset even if some do not
occur in a split. Answer-filtering pools and model-input history are configured
separately in the experiment YAML.
