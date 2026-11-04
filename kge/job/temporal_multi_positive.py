"""Train-only temporal multi-positive targets for the optional 1vsAll branch.

For a training fact (s, p, o, t), the tail targets are all train objects for
(s, p, t), and the head targets are all train subjects for (p, o, t).
The index may contain repeated facts; each distinct answer gets equal weight.
"""

from dataclasses import dataclass
from typing import Mapping, Tuple

import torch
from torch import Tensor


@dataclass(frozen=True)
class PackedPositiveTargets:
    rows: Tensor
    columns: Tensor
    counts: Tensor

    def to(self, device) -> "PackedPositiveTargets":
        return PackedPositiveTargets(
            self.rows.to(device), self.columns.to(device), self.counts.to(device)
        )


def unique_train_quadruples(quadruples: Tensor) -> Tensor:
    """Sample each exact (s,p,o,t) train fact once per epoch.

    Sorting changes row order, but the training loader shuffles every epoch.
    Distinct answers to the same temporal query remain distinct rows.
    """
    if quadruples.ndim != 2 or quadruples.shape[1] != 4:
        raise ValueError("Expected a [N,4] tensor of train quadruples")
    return torch.unique(quadruples, dim=0)


def pack_positive_targets(
    triples: Tensor,
    index: Mapping[Tuple[int, int, int], Tensor],
    key_columns: Tuple[int, int, int],
    answer_column: int,
    num_entities: int,
) -> PackedPositiveTargets:
    """Pack unique train-only answers for one prediction direction.

    Input quadruples and index keys use (s,p,o,t) column order. The returned
    tensors stay on CPU until the training job moves them to its model device.
    """
    if triples.ndim != 2 or triples.shape[1] != 4:
        raise ValueError("Expected a [batch,4] tensor of (s,p,o,t) facts")
    rows, columns, counts = [], [], []
    for row, quad in enumerate(triples.tolist()):
        key = tuple(quad[column] for column in key_columns)
        answers = index.get(key)
        if answers is None or answers.numel() == 0:
            raise ValueError(f"Training query {key} has no indexed positives")
        distinct = torch.unique(answers.long(), sorted=True)
        if distinct[0] < 0 or distinct[-1] >= num_entities:
            raise ValueError(f"Training query {key} has an out-of-range answer")
        if not torch.any(distinct == quad[answer_column]):
            raise ValueError(f"Training fact {tuple(quad)} is absent from its train index")
        values = distinct.tolist()
        rows.extend([row] * len(values))
        columns.extend(values)
        counts.append(len(values))
    return PackedPositiveTargets(
        rows=torch.tensor(rows, dtype=torch.long),
        columns=torch.tensor(columns, dtype=torch.long),
        counts=torch.tensor(counts, dtype=torch.long),
    )


def multi_positive_kl(scores: Tensor, targets: PackedPositiveTargets) -> Tensor:
    """Exact summed KL from a uniform distribution over each positive set.

    For row i with k positives, KL = logsumexp(scores_i)
    - mean(scores_i[positives]) - log(k). This equals KLDivLoss on the dense
    normalized target, while allocating only O(batch + positive-count) memory
    beyond the model's candidate score matrix.
    """
    if scores.ndim != 2 or scores.shape[0] != targets.counts.numel():
        raise ValueError("Scores and positive target rows have different batch sizes")
    if scores.shape[0] == 0 or targets.rows.numel() == 0:
        raise ValueError("Multi-positive KL needs at least one query and positive")
    if targets.rows.device != scores.device or targets.columns.device != scores.device:
        raise ValueError("Positive targets must be on the score tensor device")
    if targets.counts.device != scores.device:
        raise ValueError("Positive counts must be on the score tensor device")
    if targets.rows.numel() != targets.columns.numel():
        raise ValueError("Positive row and column coordinates differ in length")
    # Packed targets are validated on CPU when constructed; avoid a GPU sync
    # just to recheck their cardinality for every training batch.
    work_scores = scores if scores.dtype in (torch.float32, torch.float64) else scores.float()
    selected = work_scores[targets.rows, targets.columns]
    positive_sums = work_scores.new_zeros(scores.shape[0]).index_add(
        0, targets.rows, selected
    )
    counts = targets.counts.to(work_scores.dtype)
    return (
        torch.logsumexp(work_scores, dim=1)
        - positive_sums / counts
        - counts.log()
    ).sum()


def forward_with_temporal_targets(model, fn_name: str, args: tuple, kwargs: dict,
                                  targets: PackedPositiveTargets) -> Tensor:
    """Run the existing model forward with only its primary loss substituted.

    RIFT's gt_ent still identifies the sampled train fact for context
    masking. Its MLM, time-evolution, and CDC losses remain computed by the
    model itself. The original loss callback is restored even after failure.
    This deliberately supports single-process RIFT training only.
    """
    original_loss = model.loss

    def positive_loss(scores: Tensor, _sampled_answer: Tensor) -> Tensor:
        return multi_positive_kl(scores, targets)

    model.loss = positive_loss
    try:
        return model(fn_name, *args, **kwargs)
    finally:
        model.loss = original_loss
