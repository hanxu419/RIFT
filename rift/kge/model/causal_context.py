"""Query-time retrieval from a complete, time-sorted neighbor index."""

import torch


def _past_end_positions(events, offsets, entity_ids, query_time_ids):
    """First event at or after query time, within each entity's segment."""
    starts = offsets.index_select(0, entity_ids).long()
    ends = offsets.index_select(0, entity_ids + 1).long()
    left, right = starts.clone(), ends.clone()
    num_events = events.size(0)

    # Fixed iteration count avoids a GPU synchronization at each search step.
    for _ in range((num_events + 1).bit_length()):
        active = left < right
        middle = torch.div(left + right, 2, rounding_mode="floor")
        middle_time = events[middle.clamp(max=num_events - 1), 2].long()
        advance = active & (middle_time < query_time_ids)
        left = torch.where(advance, middle + 1, left)
        right = torch.where(active & ~advance, middle, right)
    return starts, left


def select_past_context(events, offsets, entity_ids, query_time_ids, max_context_size):
    """Return the latest events strictly before each query time.

    Args:
        events: [N, 3] sorted by source entity and ascending event time. Columns
            are neighbor entity, reciprocal relation, and event time.
        offsets: [num_entities + 1] boundaries for each source entity.
        entity_ids, query_time_ids: matching one-dimensional query tensors.
        max_context_size: maximum events returned per query.

    Returns:
        context [B, L, 3] (time descending), counts [B], and valid mask [B, L].
        Padding rows are zero and never count as history.
    """
    if max_context_size < 0:
        raise ValueError("max_context_size must be non-negative")
    if entity_ids.ndim != 1 or query_time_ids.ndim != 1:
        raise ValueError("entity_ids and query_time_ids must be one-dimensional")
    if entity_ids.numel() != query_time_ids.numel():
        raise ValueError("entity_ids and query_time_ids must have matching lengths")
    if (events.device != offsets.device or events.device != entity_ids.device
            or events.device != query_time_ids.device):
        raise ValueError("Neighbor index and query IDs must be on the same device")

    batch_size = entity_ids.numel()
    length = int(max_context_size)
    if length == 0 or events.size(0) == 0:
        empty = torch.zeros((batch_size, length, 3), dtype=torch.long, device=entity_ids.device)
        mask = torch.zeros((batch_size, length), dtype=torch.bool, device=entity_ids.device)
        return empty, mask.sum(dim=1), mask

    entity_ids = entity_ids.long()
    query_time_ids = query_time_ids.long()
    num_events = events.size(0)
    starts, left = _past_end_positions(events, offsets, entity_ids, query_time_ids)

    positions = left.unsqueeze(1) - 1 - torch.arange(length, device=entity_ids.device)
    valid = positions >= starts.unsqueeze(1)
    context = events[positions.clamp(min=0, max=num_events - 1)].long()
    context = torch.where(valid.unsqueeze(-1), context, torch.zeros_like(context))
    return context, valid.sum(dim=1), valid


def select_dual_past_context(
    events, offsets, relation_positions, relation_keys, relation_offsets,
    entity_ids, query_time_ids, history_relation_ids, num_relations,
):
    """Select 16 latest events, then 16 older relation-matched events.

    Any shortage in the second group is filled by the latest remaining older
    events. The final 32 slots are in descending time/index order, with zero
    padding. `history_relation_ids` uses the encoding stored in `events`, not
    necessarily the query embedding's relation ID: outgoing (tail-query)
    events have p+R and incoming (head-query) events have p.
    """
    tensors = (
        events, offsets, relation_positions, relation_keys, relation_offsets,
        entity_ids, query_time_ids, history_relation_ids,
    )
    if any(t.device != events.device for t in tensors):
        raise ValueError("Neighbor index and query IDs must be on the same device")
    if any(t.ndim != 1 for t in (entity_ids, query_time_ids, history_relation_ids)):
        raise ValueError("Query IDs must be one-dimensional")
    if not (entity_ids.numel() == query_time_ids.numel() == history_relation_ids.numel()):
        raise ValueError("Query IDs must have matching lengths")
    if num_relations <= 0:
        raise ValueError("num_relations must be positive")

    batch_size = entity_ids.numel()
    device = events.device
    if events.size(0) == 0:
        empty = torch.zeros((batch_size, 32, 3), dtype=torch.long, device=device)
        mask = torch.zeros((batch_size, 32), dtype=torch.bool, device=device)
        return empty, mask.sum(dim=1), mask

    starts, past_end = _past_end_positions(
        events, offsets, entity_ids.long(), query_time_ids.long()
    )
    recent = past_end[:, None] - 1 - torch.arange(16, device=device)
    recent = torch.where(recent >= starts[:, None], recent, -1)
    older_end = torch.maximum(starts, past_end - 16)

    # Sparse pair lookup, then batched lower-bound search for the first
    # relation event whose original position is >= older_end.
    keys = entity_ids.long() * (2 * num_relations) + history_relation_ids.long()
    pair_index = torch.searchsorted(relation_keys, keys)
    safe_pair_index = pair_index.clamp(max=relation_keys.numel() - 1)
    found = ((pair_index < relation_keys.numel()) &
             (relation_keys[safe_pair_index] == keys))
    pair_start = torch.where(found, relation_offsets[safe_pair_index], 0)
    pair_end = torch.where(found, relation_offsets[safe_pair_index + 1], 0)
    left, right = pair_start.clone(), pair_end.clone()
    event_count = events.size(0)
    for _ in range((event_count + 1).bit_length()):
        active = left < right
        middle = torch.div(left + right, 2, rounding_mode="floor")
        event_pos = relation_positions[middle.clamp(max=event_count - 1)].long()
        advance = active & (event_pos < older_end)
        left = torch.where(advance, middle + 1, left)
        right = torch.where(active & ~advance, middle, right)

    match_idx = left[:, None] - 1 - torch.arange(16, device=device)
    match_valid = match_idx >= pair_start[:, None]
    matched = relation_positions[match_idx.clamp(min=0, max=event_count - 1)].long()
    matched = torch.where(match_valid, matched, -1)

    # At most 16 matches can occur in the latest 32 older events. Reading 32
    # candidates therefore guarantees enough nonmatching backfill, if present.
    older = older_end[:, None] - 1 - torch.arange(32, device=device)
    older_valid = older >= starts[:, None]
    duplicate = (older[:, :, None] == matched[:, None, :]).any(dim=2)
    fill_valid = older_valid & ~duplicate
    fill_rank = fill_valid.long().cumsum(dim=1)
    fill_need = 16 - match_valid.long().sum(dim=1, keepdim=True)
    filled = torch.where(fill_valid & (fill_rank <= fill_need), older, -1)

    positions = torch.cat((recent, matched, filled), dim=1).sort(
        dim=1, descending=True
    ).values[:, :32]
    valid = positions >= 0
    context = events[positions.clamp(min=0)].long()
    context = torch.where(valid[:, :, None], context, torch.zeros_like(context))
    return context, valid.sum(dim=1), valid


def select_time_stratified_past_context(events, offsets, entity_ids, query_time_ids):
    """Select 16 latest events plus up to 16 older, relation-agnostic events.

    The older time span is divided by 16 equally spaced timestamp anchors.
    For each anchor, select the latest event at or before it. Repeated picks
    are removed and vacant slots are filled with the latest remaining older
    events. This keeps the total budget at 32, preserves strict query-time
    causality, and separates broader temporal reach from relation matching.
    """
    if entity_ids.ndim != 1 or query_time_ids.ndim != 1:
        raise ValueError("entity_ids and query_time_ids must be one-dimensional")
    if entity_ids.numel() != query_time_ids.numel():
        raise ValueError("entity_ids and query_time_ids must have matching lengths")
    if events.device != offsets.device or events.device != entity_ids.device or events.device != query_time_ids.device:
        raise ValueError("Neighbor index and query IDs must be on the same device")

    batch_size = entity_ids.numel()
    device = events.device
    event_count = events.size(0)
    if event_count == 0:
        empty = torch.zeros((batch_size, 32, 3), dtype=torch.long, device=device)
        mask = torch.zeros((batch_size, 32), dtype=torch.bool, device=device)
        return empty, mask.sum(dim=1), mask

    starts, past_end = _past_end_positions(
        events, offsets, entity_ids.long(), query_time_ids.long()
    )
    recent = past_end[:, None] - 1 - torch.arange(16, device=device)
    recent = torch.where(recent >= starts[:, None], recent, -1)
    older_end = torch.maximum(starts, past_end - 16)
    older_count = older_end - starts
    has_older = older_count > 0

    oldest_time = events[starts.clamp(max=event_count - 1), 2].long()
    newest_time = events[(older_end - 1).clamp(min=0, max=event_count - 1), 2].long()
    span = (newest_time - oldest_time).clamp(min=0)
    anchors = newest_time[:, None] - torch.div(
        torch.arange(16, device=device)[None, :] * span[:, None],
        15, rounding_mode="floor",
    )

    # Upper bound of each timestamp anchor inside its entity's older segment.
    left = starts[:, None].expand(-1, 16).clone()
    right = older_end[:, None].expand(-1, 16).clone()
    for _ in range((event_count + 1).bit_length()):
        active = left < right
        middle = torch.div(left + right, 2, rounding_mode="floor")
        middle_time = events[middle.clamp(max=event_count - 1), 2].long()
        advance = active & (middle_time <= anchors)
        left = torch.where(advance, middle + 1, left)
        right = torch.where(active & ~advance, middle, right)
    selected = left - 1
    selected = torch.where(has_older[:, None] & (selected >= starts[:, None]), selected, -1)

    # Several anchors may hit the same event, especially at equal timestamps.
    ordinal = torch.arange(16, device=device)
    seen_before = (
        (selected[:, :, None] == selected[:, None, :])
        & (ordinal[None, None, :] < ordinal[None, :, None])
    ).any(dim=2)
    selected = torch.where(seen_before, -1, selected)

    # Reading the 32 latest older positions guarantees enough unused events
    # for any shortfall among the 16 anchor picks.
    candidates = older_end[:, None] - 1 - torch.arange(32, device=device)
    candidate_valid = candidates >= starts[:, None]
    duplicate = (candidates[:, :, None] == selected[:, None, :]).any(dim=2)
    available = candidate_valid & ~duplicate
    available_rank = available.long().cumsum(dim=1)
    need = 16 - (selected >= 0).long().sum(dim=1, keepdim=True)
    backfill = torch.where(available & (available_rank <= need), candidates, -1)

    older = torch.cat((selected, backfill), dim=1).sort(dim=1, descending=True).values[:, :16]
    positions = torch.cat((recent, older), dim=1).sort(dim=1, descending=True).values
    valid = positions >= 0
    context = events[positions.clamp(min=0, max=event_count - 1)].long()
    context = torch.where(valid[:, :, None], context, torch.zeros_like(context))
    return context, valid.sum(dim=1), valid
