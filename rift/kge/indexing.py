import torch
from collections import defaultdict, OrderedDict
try:
    import numba
except ImportError:
    class _NumbaFallback:
        @staticmethod
        def njit(*args, **kwargs):
            # Fallback decorator: run regular Python implementation without JIT.
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]

            def _decorator(func):
                return func

            return _decorator

    numba = _NumbaFallback()
import numpy as np
import networkx as nx




# -----------------------------------------------------------------------------
# Neighbor indexing utilities (time-aware ordering & stratified time-bucket sampling)
# -----------------------------------------------------------------------------
def _cfg_get_any(dataset, keys, default=None):
    """Best-effort config getter.

    The KGE config system differs across forks; this helper tries multiple key
    candidates and gracefully falls back to `default` if a key is missing.
    """
    cfg = getattr(dataset, "config", None)
    if cfg is None:
        return default
    for k in keys:
        try:
            # common signature: cfg.get(key)
            v = cfg.get(k)
            if v is not None:
                return v
        except Exception:
            pass
        try:
            # alternative: cfg.get(key, default)
            v = cfg.get(k, default)
            if v is not None:
                return v
        except Exception:
            pass
    return default


def _cfg_bool(dataset, keys, default=False):
    v = _cfg_get_any(dataset, keys, default)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y", "on", "enable", "enabled")
    return bool(v)


def _cfg_int(dataset, keys, default):
    v = _cfg_get_any(dataset, keys, default)
    try:
        return int(v)
    except Exception:
        return int(default)


def _cfg_float(dataset, keys, default):
    v = _cfg_get_any(dataset, keys, default)
    try:
        return float(v)
    except Exception:
        return float(default)


def _cfg_str(dataset, keys, default):
    v = _cfg_get_any(dataset, keys, default)
    if v is None:
        return default
    return str(v)


def _get_rng(dataset):
    """Derive a deterministic RNG if a seed is available in config."""
    explicit_seed = _cfg_get_any(dataset, ["rift.neighbor.index_seed"], None)
    if explicit_seed is not None:
        try:
            if int(explicit_seed) >= 0:
                return np.random.default_rng(int(explicit_seed))
        except (TypeError, ValueError):
            pass
    seed = _cfg_get_any(
        dataset,
        keys=[
            "seed",
            "job.seed",
            "train.seed",
            "random_seed",
            "kge.seed",
            "rift.seed",
        ],
        default=None,
    )
    try:
        if seed is None:
            return np.random.default_rng()
        return np.random.default_rng(int(seed))
    except Exception:
        return np.random.default_rng()


def _stable_time_sort_idx(times_np: np.ndarray, rng: np.random.Generator, desc: bool):
    """Sort by time with randomized tie-breaking (stable across a run)."""
    n = len(times_np)
    if n <= 1:
        return np.arange(n, dtype=np.int64)
    perm = rng.permutation(n)
    t = times_np[perm]
    # argsort is stable in numpy for kind='mergesort'
    order = np.argsort(t, kind="mergesort")
    if desc:
        order = order[::-1]
    return perm[order]


def _stratified_time_bucket_sample(
    times_np: np.ndarray,
    take: int,
    rng: np.random.Generator,
    num_bins: int = 4,
    min_per_bin: int = 1,
    interleave: bool = True,
    desc: bool = True,
):
    """Return indices sampled across time buckets.

    Buckets are created by time *quantiles* (robust across skewed time distributions).
    Then we allocate samples roughly proportional to bucket sizes with a floor of
    `min_per_bin` for non-empty buckets (as long as budget permits).
    """
    n = len(times_np)
    if n == 0 or take <= 0:
        return np.empty((0,), dtype=np.int64)
    take = min(take, n)
    if num_bins <= 1 or n == 1:
        return _stable_time_sort_idx(times_np, rng, desc=desc)[:take]

    # Create quantile boundaries
    t_sorted = np.sort(times_np)
    qs = np.linspace(0.0, 1.0, num_bins + 1)
    bounds = np.quantile(t_sorted, qs)
    # Ensure bounds are non-decreasing (quantile can repeat if many ties)
    # We'll define bins as [b[i], b[i+1]] with right-closed on last bin.
    bins = []
    for i in range(num_bins):
        lo = bounds[i]
        hi = bounds[i + 1]
        if i == num_bins - 1:
            mask = (times_np >= lo) & (times_np <= hi)
        else:
            mask = (times_np >= lo) & (times_np < hi)
        idx = np.nonzero(mask)[0]
        if idx.size > 0:
            bins.append(idx)

    if len(bins) == 0:
        return _stable_time_sort_idx(times_np, rng, desc=desc)[:take]

    # Allocate counts
    sizes = np.array([len(b) for b in bins], dtype=np.int64)
    total = sizes.sum()
    # base proportional allocation
    raw = sizes / max(total, 1) * take
    counts = np.floor(raw).astype(np.int64)
    # enforce minimum per non-empty bin if possible
    if take >= len(bins) * min_per_bin:
        counts = np.maximum(counts, min_per_bin)
    # fix total via remainder distribution
    diff = int(take - counts.sum())
    if diff > 0:
        frac = raw - np.floor(raw)
        order = np.argsort(frac)[::-1]
        for j in order:
            if diff <= 0:
                break
            counts[j] += 1
            diff -= 1
    elif diff < 0:
        # remove from bins with the largest counts first, but keep >=1 where possible
        order = np.argsort(counts)[::-1]
        for j in order:
            if diff >= 0:
                break
            if counts[j] > 1:
                counts[j] -= 1
                diff += 1

    # Sample per bin (without replacement)
    selected_per_bin = []
    for b, c in zip(bins, counts.tolist()):
        if c <= 0:
            selected_per_bin.append(np.empty((0,), dtype=np.int64))
            continue
        if c >= len(b):
            chosen = b.copy()
            rng.shuffle(chosen)
        else:
            chosen = rng.choice(b, size=c, replace=False)
        selected_per_bin.append(chosen)

    # Combine
    if not interleave:
        combined = np.concatenate(selected_per_bin, axis=0)
        # For final ordering, sort by time (with tie-break) if requested
        return combined[:take]

    # Interleave across bins; prefer newest-first (desc) by default
    # Determine bin recency by mean time
    bin_mean_t = np.array([times_np[b].mean() if len(b) else -1 for b in selected_per_bin])
    bin_order = np.argsort(bin_mean_t)
    if desc:
        bin_order = bin_order[::-1]

    ptr = [0] * len(selected_per_bin)
    out = []
    while len(out) < take:
        progressed = False
        for bi in bin_order:
            arr = selected_per_bin[bi]
            if ptr[bi] < len(arr):
                out.append(arr[ptr[bi]])
                ptr[bi] += 1
                progressed = True
                if len(out) >= take:
                    break
        if not progressed:
            break
    return np.array(out, dtype=np.int64)


def _order_neighbors(dataset, neighbor_np, rel_np, time_np, max_neighbor_num, rng):
    """Apply ordering/sampling policy configured via neighbor.order."""
    order = _cfg_str(
        dataset,
        keys=[
            "rift.neighbor.index_order",
            "model.neighbor.index_order",
            "neighbor.index_order",
            "rift.neighbor.order",
            "model.neighbor.order",
            "neighbor.order",
        ],
        default="random",
    ).strip().lower()

    take = min(max_neighbor_num, len(neighbor_np))
    if take <= 0:
        return neighbor_np[:0], rel_np[:0], time_np[:0]

    if order in ("random", "rand"):
        idx = rng.permutation(len(neighbor_np))[:take]
    elif order in ("by_time_desc", "time_desc", "desc"):
        idx = _stable_time_sort_idx(time_np, rng, desc=True)[:take]
    elif order in ("by_time_asc", "time_asc", "asc"):
        idx = _stable_time_sort_idx(time_np, rng, desc=False)[:take]
    elif order in ("stratified", "time_bucket", "time_bucketed", "bucket"):
        num_bins = _cfg_int(
            dataset,
            keys=[
    "rift.neighbor.index_time_bucket_bins",
    "rift.neighbor.index_stratified_bins",
    "neighbor.index_time_bucket_bins",
    "neighbor.index_stratified_bins",
                "rift.neighbor.time_bucket_bins",
                "rift.neighbor.stratified_bins",
                "neighbor.time_bucket_bins",
                "neighbor.stratified_bins",
            ],
            default=4,
        )
        min_per_bin = _cfg_int(
            dataset,
            keys=[
    "rift.neighbor.index_time_bucket_min_per_bin",
    "rift.neighbor.index_stratified_min_per_bin",
    "neighbor.index_time_bucket_min_per_bin",
    "neighbor.index_stratified_min_per_bin",
                "rift.neighbor.time_bucket_min_per_bin",
                "rift.neighbor.stratified_min_per_bin",
                "neighbor.time_bucket_min_per_bin",
                "neighbor.stratified_min_per_bin",
            ],
            default=1,
        )
        interleave = _cfg_bool(
            dataset,
            keys=[
    "rift.neighbor.index_time_bucket_interleave",
    "rift.neighbor.index_stratified_interleave",
    "neighbor.index_time_bucket_interleave",
    "neighbor.index_stratified_interleave",
                "rift.neighbor.time_bucket_interleave",
                "rift.neighbor.stratified_interleave",
                "neighbor.time_bucket_interleave",
                "neighbor.stratified_interleave",
            ],
            default=True,
        )
        desc = _cfg_bool(
            dataset,
            keys=[
    "rift.neighbor.index_time_bucket_desc",
    "rift.neighbor.index_stratified_desc",
    "neighbor.index_time_bucket_desc",
    "neighbor.index_stratified_desc",
                "rift.neighbor.time_bucket_desc",
                "rift.neighbor.stratified_desc",
                "neighbor.time_bucket_desc",
                "neighbor.stratified_desc",
            ],
            default=True,
        )
        idx = _stratified_time_bucket_sample(
            time_np, take=take, rng=rng, num_bins=num_bins, min_per_bin=min_per_bin, interleave=interleave, desc=desc
        )
    else:
        # unknown -> keep legacy behavior (random)
        idx = rng.permutation(len(neighbor_np))[:take]

    return neighbor_np[idx], rel_np[idx], time_np[idx]

def _group_by(keys, values) -> dict:
    """Group values by keys.

    :param keys: list of keys
    :param values: list of values
    A key value pair i is defined by (key_list[i], value_list[i]).
    :return: OrderedDict where key value pairs have been grouped by key.

     """
    result = defaultdict(list)
    for key, value in zip(keys.tolist(), values.tolist()):
        result[tuple(key)].append(value)
    for key, value in result.items():
        result[key] = torch.IntTensor(sorted(value))
    return OrderedDict(result)


def index_KvsAll(dataset: "Dataset", split: str, key: str):
    """Return an index for the triples in split (''train'', ''valid'', ''test'')
    from the specified key (''sp'' or ''po'' or ''so'') to the indexes of the
    remaining constituent (''o'' or ''s'' or ''p'' , respectively.)

    The index maps from `tuple' to `torch.LongTensor`.

    The index is cached in the provided dataset under name `{split}_sp_to_o` or
    `{split}_po_to_s`, or `{split}_so_to_p`. If this index is already present, does not
    recompute it.

    """
    value = None
    if key == "sp":
        key_cols = [0, 1]
        value_column = 2
        value = "o"
    elif key == "po":
        key_cols = [1, 2]
        value_column = 0
        value = "s"
    elif key == "so":
        key_cols = [0, 2]
        value_column = 1
        value = "p"
    else:
        raise ValueError()

    name = split + "_" + key + "_to_" + value
    if not dataset._indexes.get(name):
        triples = dataset.split(split)
        dataset._indexes[name] = _group_by(
            triples[:, key_cols], triples[:, value_column]
        )

    dataset.config.log(
        "{} distinct {} pairs in {}".format(len(dataset._indexes[name]), key, split),
        prefix="  ",
    )

    return dataset._indexes.get(name)


def index_temporal_KvsAll(dataset: "Dataset", split: str, key: str):
    """Index true answers at the query timestamp for temporal filtered ranking.

    These indexes are separate from the time-agnostic KvsAll training indexes so
    that older checkpoints and non-temporal jobs retain their original protocol.
    Quadruples in this project have the column order (s, p, o, t).
    """
    if key == "spt":
        key_cols, value_column, value = [0, 1, 3], 2, "o"
    elif key == "pot":
        key_cols, value_column, value = [1, 2, 3], 0, "s"
    else:
        raise ValueError(f"Unsupported temporal index key: {key}")
    name = f"{split}_{key}_to_{value}"
    if name not in dataset._indexes:
        quadruples = dataset.split(split)
        if quadruples.shape[1] != 4:
            raise ValueError("Temporal filtering requires (s, p, o, t) quadruples")
        dataset._indexes[name] = _group_by(
            quadruples[:, key_cols], quadruples[:, value_column]
        )
    dataset.config.log(
        f"{len(dataset._indexes[name])} distinct {key} keys in {split}", prefix="  "
    )
    return dataset._indexes[name]


def index_causal_neighbor(dataset):
    """Build a complete, time-sorted train-only neighbor index.

    Returns (events, offsets). Events are sorted by source entity, then time,
    with columns (neighbor entity, reciprocal relation, time). For entity e,
    its events occupy events[offsets[e]:offsets[e+1]]. No global top-K cap is
    applied: the model selects the latest strictly past events for each query.
    """
    name = "causal_neighbor"
    if name not in dataset._indexes:
        quadruples = dataset.split("train")
        if quadruples.shape[1] != 4:
            raise ValueError("Causal neighbors require (s, p, o, t) quadruples")
        q = quadruples.detach().cpu().numpy().astype(np.int64, copy=False)
        s, p, o, t = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        source = np.concatenate((s, o))
        neighbor = np.concatenate((o, s))
        relation = np.concatenate((p + dataset.num_relations(), p))
        time = np.concatenate((t, t))
        num_entities = dataset.num_entities()
        if np.any(source < 0) or np.any(source >= num_entities):
            raise ValueError("Train quadruples contain out-of-range entity IDs")
        # Stable sorting preserves input order among events at the same time.
        sort_key = source * max(dataset.num_times(), 1) + time
        order = np.argsort(sort_key, kind="stable")
        events_np = np.stack((neighbor[order], relation[order], time[order]), axis=1)
        events = torch.from_numpy(events_np.astype(np.int32, copy=False))
        counts = np.bincount(source[order], minlength=num_entities)
        offsets_np = np.empty(num_entities + 1, dtype=np.int64)
        offsets_np[0] = 0
        np.cumsum(counts, out=offsets_np[1:])
        offsets = torch.from_numpy(offsets_np)
        dataset._indexes[name] = (events, offsets)
    dataset.config.log(
        f"Causal neighbor index: {dataset._indexes[name][0].shape[0]} directed train events",
        prefix="  ",
    )
    return dataset._indexes[name]


def index_causal_neighbor_relation(dataset):
    """Add a compact relation lookup to the complete causal neighbor index.

    The returned event positions are grouped by (source entity, encoded edge
    relation) and ascending original event position. Looking up an older
    same-relation event then costs O(log N), without materializing a dense
    [batch, all-history] tensor or duplicating event triples.
    """
    name = "causal_neighbor_relation"
    if name not in dataset._indexes:
        events, offsets = index_causal_neighbor(dataset)
        event_count = events.size(0)
        if event_count:
            counts = np.diff(offsets.numpy())
            sources = np.repeat(np.arange(dataset.num_entities(), dtype=np.int64), counts)
            pair_keys = sources * (2 * dataset.num_relations()) + events[:, 1].numpy()
            # The original event index is already ordered by time for each
            # source. Stable key sorting preserves that order within each pair.
            order = np.argsort(pair_keys, kind="stable")
            keys, first = np.unique(pair_keys[order], return_index=True)
            positions = torch.from_numpy(order.astype(np.int64, copy=False))
            pair_keys_tensor = torch.from_numpy(keys.astype(np.int64, copy=False))
            pair_offsets = torch.from_numpy(
                np.concatenate((first, np.array([event_count], dtype=np.int64)))
            )
        else:
            positions = torch.empty(0, dtype=torch.long)
            pair_keys_tensor = torch.empty(0, dtype=torch.long)
            pair_offsets = torch.zeros(1, dtype=torch.long)
        dataset._indexes[name] = (
            events, offsets, positions, pair_keys_tensor, pair_offsets
        )
    return dataset._indexes[name]


def index_KvsAll_to_torch(index):
    """Convert `index_KvsAll` indexes to pytorch tensors.

    Returns an nx2 keys tensor (rows = keys), an offset vector
    (row = starting offset in values for corresponding key),
    a values vector (entries correspond to values of original
    index)

    Afterwards, it holds:
        index[keys[i]] = values[offsets[i]:offsets[i+1]]
    """
    keys = torch.tensor(list(index.keys()), dtype=torch.int)
    values = torch.cat(list(index.values()))
    offsets = torch.cumsum(
        torch.tensor([0] + list(map(len, index.values())), dtype=torch.int), 0
    )
    return keys, values, offsets


def index_frequency_percent(dataset):
    name = "fre"
    if not dataset._indexes.get(name):
        fre = []
        entities_fre = {}
        train_triples = dataset.split('train')
        for i in range(dataset.num_entities()):
            entities_fre[i] = 0
        for tri in train_triples:
            s, p, t, o = tri.tolist()
            if s in entities_fre:
                entities_fre[s] += 1
        for i in range(dataset.num_entities()):
            fre.append(entities_fre[i] / dataset.num_entities())
        dataset._indexes[name] = fre
    dataset.config.log("Entities_fre index finished", prefix="  ")
    return dataset._indexes[name]


def index_neighbor_multidig(dataset):
    """Build neighbor index with configurable ordering / sampling.

    Output format (backward compatible):
        dataset._indexes["neighbor"] = (all_neighbor, all_neighbor_num)
    where:
        all_neighbor.shape = (num_entities, 3, max_neighbor_num)
        all_neighbor[:,0,:] = neighbor entity ids
        all_neighbor[:,1,:] = neighbor relation ids (incl. reciprocal encoding)
        all_neighbor[:,2,:] = neighbor time ids

    Config keys (best-effort; multiple aliases supported):
        - rift.neighbor.order / neighbor.order:
            random | by_time_desc | by_time_asc | stratified
        - rift.neighbor.stratified_bins / neighbor.stratified_bins: int (default 4)
        - rift.neighbor.stratified_min_per_bin / neighbor.stratified_min_per_bin: int (default 1)
        - rift.neighbor.stratified_interleave / neighbor.stratified_interleave: bool (default True)
        - rift.neighbor.stratified_desc / neighbor.stratified_desc: bool (default True)

    Notes:
        - Ordering is applied *at index-build time* and therefore cached per run.
        - For stochastic policies (random/stratified), randomness is controlled by
          config seed if present; otherwise non-deterministic.
    """
    name = "neighbor"
    if not dataset._indexes.get(name):
        train_triples = dataset.split("train")
        G = nx.MultiDiGraph()
        for tri in train_triples:
            # Note: this project uses (s, p, o, t)
            s, p, o, t = tri.tolist()
            G.add_node(s)
            G.add_node(o)
            G.add_edge(s, o, type=p, time=t)

        max_neighbor_num = _cfg_int(
            dataset, ["rift.neighbor.index_max_neighbors"], 128
        )
        if max_neighbor_num < 1:
            raise ValueError("rift.neighbor.index_max_neighbors must be positive")
        all_neighbor = torch.zeros(
            (dataset.num_entities(), 3, max_neighbor_num), dtype=torch.long
        )
        all_neighbor_num = torch.zeros(dataset.num_entities(), dtype=torch.long)

        # collect neighbor lists
        neighbor_all = [[] for _ in range(dataset.num_entities())]
        neighbor_edge_types_all = [[] for _ in range(dataset.num_entities())]
        neighbor_edge_times_all = [[] for _ in range(dataset.num_entities())]

        rng = _get_rng(dataset)

        for s, o, data in G.edges(data=True):
            # undirected view: add both endpoints to each other's neighbor set
            neighbor_all[s].append(o)
            neighbor_all[o].append(s)

            # reciprocal encoding: from source side use p+R, from target side use p
            neighbor_edge_types_all[s].append(int(data["type"]) + dataset.num_relations())
            neighbor_edge_times_all[s].append(int(data["time"]))

            neighbor_edge_types_all[o].append(int(data["type"]))
            neighbor_edge_times_all[o].append(int(data["time"]))

        # Apply ordering/sampling per entity
        used_order = _cfg_str(
            dataset,
            keys=[
                "rift.neighbor.index_order",
                "model.neighbor.index_order",
                "neighbor.index_order",
                "rift.neighbor.order",
                "model.neighbor.order",
                "neighbor.order",
            ],
            default="random",
        )
        dataset.config.log(
            f"Building neighbor index with neighbor.order={used_order}",
            prefix="  ",
        )

        for s in range(dataset.num_entities()):
            if s not in G:
                continue
            n = len(neighbor_all[s])
            if n <= 0:
                continue

            neighbor_np = np.asarray(neighbor_all[s], dtype=np.int64)
            rel_np = np.asarray(neighbor_edge_types_all[s], dtype=np.int64)
            time_np = np.asarray(neighbor_edge_times_all[s], dtype=np.int64)

            # Apply configured ordering / time-bucket sampling
            neighbor_np, rel_np, time_np = _order_neighbors(
                dataset, neighbor_np, rel_np, time_np, max_neighbor_num, rng
            )

            k = len(neighbor_np)
            all_neighbor[s, 0, 0:k] = torch.from_numpy(neighbor_np.astype(np.int64))
            all_neighbor[s, 1, 0:k] = torch.from_numpy(rel_np.astype(np.int64))
            all_neighbor[s, 2, 0:k] = torch.from_numpy(time_np.astype(np.int64))
            all_neighbor_num[s] = k

        dataset._indexes[name] = (all_neighbor, all_neighbor_num)

    dataset.config.log("Neighbors index finished", prefix="  ")
    return dataset._indexes.get(name)



def index_neighbor_dig(dataset):
    """Directed neighbor index with configurable ordering / sampling.

    Same output format as index_neighbor_multidig, but uses a DiGraph and
    predecessors/successors decomposition.
    """
    name = "neighbor"
    if not dataset._indexes.get(name):
        train_triples = dataset.split("train")
        G = nx.DiGraph()
        for tri in train_triples:
            s, p, o, t = tri.tolist()
            G.add_node(s)
            G.add_node(o)
            G.add_edge(s, o, type=p, time=t)

        max_neighbor_num = 300
        all_neighbor = torch.zeros(
            (dataset.num_entities(), 3, max_neighbor_num), dtype=torch.long
        )
        all_neighbor_num = torch.zeros(dataset.num_entities(), dtype=torch.long)

        rng = _get_rng(dataset)
        used_order = _cfg_str(
            dataset,
            keys=[
                "rift.neighbor.index_order",
                "model.neighbor.index_order",
                "neighbor.index_order",
                "rift.neighbor.order",
                "model.neighbor.order",
                "neighbor.order",
            ],
            default="random",
        )
        dataset.config.log(
            f"Building directed neighbor index with neighbor.order={used_order}",
            prefix="  ",
        )

        for s in range(dataset.num_entities()):
            if s not in G:
                continue
            suc = list(G.successors(s))
            pre = list(G.predecessors(s))

            suc_edge_types = [
                int(G.get_edge_data(s, v)["type"]) + dataset.num_relations() for v in suc
            ]
            pre_edge_types = [int(G.get_edge_data(v, s)["type"]) for v in pre]
            suc_edge_times = [int(G.get_edge_data(s, v)["time"]) for v in suc]
            pre_edge_times = [int(G.get_edge_data(v, s)["time"]) for v in pre]

            neighbor_np = np.asarray(suc + pre, dtype=np.int64)
            rel_np = np.asarray(suc_edge_types + pre_edge_types, dtype=np.int64)
            time_np = np.asarray(suc_edge_times + pre_edge_times, dtype=np.int64)

            if len(neighbor_np) == 0:
                continue

            neighbor_np, rel_np, time_np = _order_neighbors(
                dataset, neighbor_np, rel_np, time_np, max_neighbor_num, rng
            )

            k = len(neighbor_np)
            all_neighbor[s, 0, 0:k] = torch.from_numpy(neighbor_np.astype(np.int64))
            all_neighbor[s, 1, 0:k] = torch.from_numpy(rel_np.astype(np.int64))
            all_neighbor[s, 2, 0:k] = torch.from_numpy(time_np.astype(np.int64))
            all_neighbor_num[s] = k

        dataset._indexes[name] = (all_neighbor, all_neighbor_num)

    dataset.config.log("Neighbors index finished", prefix="  ")
    return dataset._indexes.get(name)



def index_relation_types(dataset):
    """Classify relations into 1-N, M-1, 1-1, M-N.

    According to Bordes et al. "Translating embeddings for modeling multi-relational
    data.", NIPS13.

    Adds index `relation_types` with list that maps relation index to ("1-N", "M-1",
    "1-1", "M-N").

    """
    if "relation_types" not in dataset._indexes:
        # 2nd dim: num_s, num_distinct_po, num_o, num_distinct_so, is_M, is_N
        relation_stats = torch.zeros((dataset.num_relations(), 6))
        for index, p in [
            (dataset.index("train_sp_to_o"), 1),
            (dataset.index("train_po_to_s"), 0),
        ]:
            for prefix, labels in index.items():
                relation_stats[prefix[p], 0 + p * 2] = relation_stats[
                    prefix[p], 0 + p * 2
                ] + len(labels)
                relation_stats[prefix[p], 1 + p * 2] = (
                    relation_stats[prefix[p], 1 + p * 2] + 1.0
                )
        relation_stats[:, 4] = (relation_stats[:, 0] / relation_stats[:, 1]) > 1.5
        relation_stats[:, 5] = (relation_stats[:, 2] / relation_stats[:, 3]) > 1.5
        relation_types = []
        for i in range(dataset.num_relations()):
            relation_types.append(
                "{}-{}".format(
                    "1" if relation_stats[i, 4].item() == 0 else "M",
                    "1" if relation_stats[i, 5].item() == 0 else "N",
                )
            )

        dataset._indexes["relation_types"] = relation_types

    return dataset._indexes["relation_types"]


def index_relations_per_type(dataset):
    if "relations_per_type" not in dataset._indexes:
        relations_per_type = {}
        for i, k in enumerate(dataset.index("relation_types")):
            relations_per_type.setdefault(k, set()).add(i)
        dataset._indexes["relations_per_type"] = relations_per_type
    else:
        relations_per_type = dataset._indexes["relations_per_type"]

    dataset.config.log("Loaded relation index")
    for k, relations in relations_per_type.items():
        dataset.config.log(
            "{} relations of type {}".format(len(relations), k), prefix="  "
        )

    return relations_per_type


def index_frequency_percentiles(dataset, recompute=False):
    """
    :return: dictionary mapping from
    {
        'subject':
        {25%, 50%, 75%, top} -> set of entities
        'relations':
        {25%, 50%, 75%, top} -> set of relations
        'object':
        {25%, 50%, 75%, top} -> set of entities
    }
    """
    if "frequency_percentiles" in dataset._indexes and not recompute:
        return
    subject_stats = torch.zeros((dataset.num_entities(), 1))
    relation_stats = torch.zeros((dataset.num_relations(), 1))
    object_stats = torch.zeros((dataset.num_entities(), 1))
    for (s, p, o) in dataset.split("train"):
        subject_stats[s] += 1
        relation_stats[p] += 1
        object_stats[o] += 1
    result = dict()
    for arg, stats, num in [
        (
            "subject",
            [
                i
                for i, j in list(
                    sorted(enumerate(subject_stats.tolist()), key=lambda x: x[1])
                )
            ],
            dataset.num_entities(),
        ),
        (
            "relation",
            [
                i
                for i, j in list(
                    sorted(enumerate(relation_stats.tolist()), key=lambda x: x[1])
                )
            ],
            dataset.num_relations(),
        ),
        (
            "object",
            [
                i
                for i, j in list(
                    sorted(enumerate(object_stats.tolist()), key=lambda x: x[1])
                )
            ],
            dataset.num_entities(),
        ),
    ]:
        for percentile, (begin, end) in [
            ("25%", (0.0, 0.25)),
            ("50%", (0.25, 0.5)),
            ("75%", (0.5, 0.75)),
            ("top", (0.75, 1.0)),
        ]:
            if arg not in result:
                result[arg] = dict()
            result[arg][percentile] = set(stats[int(begin * num) : int(end * num)])
    dataset._indexes["frequency_percentiles"] = result


class IndexWrapper:
    """Wraps a call to an index function so that it can be pickled"""

    def __init__(self, fun, **kwargs):
        self.fun = fun
        self.kwargs = kwargs

    def __call__(self, dataset: "Dataset", **kwargs):
        self.fun(dataset, **self.kwargs)


def _invert_ids(dataset, obj: str):
    if not f"{obj}_id_to_index" in dataset._indexes:
        ids = dataset.load_map(f"{obj}_ids")
        inv = {v: k for k, v in enumerate(ids)}
        dataset._indexes[f"{obj}_id_to_index"] = inv
    else:
        inv = dataset._indexes[f"{obj}_id_to_index"]
    dataset.config.log(f"Indexed {len(inv)} {obj} ids", prefix="  ")


def create_default_index_functions(dataset: "Dataset"):
    for split in dataset.files_of_type("triples"):
        for key, value in [("sp", "o"), ("po", "s"), ("so", "p")]:
            # self assignment needed to capture the loop var
            dataset.index_functions[f"{split}_{key}_to_{value}"] = IndexWrapper(
                index_KvsAll, split=split, key=key
            )
        for key, value in [("spt", "o"), ("pot", "s")]:
            dataset.index_functions[f"{split}_{key}_to_{value}"] = IndexWrapper(
                index_temporal_KvsAll, split=split, key=key
            )
    dataset.index_functions["neighbor"] = index_neighbor_multidig
    dataset.index_functions["causal_neighbor"] = index_causal_neighbor
    dataset.index_functions["causal_neighbor_relation"] = index_causal_neighbor_relation
    dataset.index_functions["relation_types"] = index_relation_types
    dataset.index_functions["relations_per_type"] = index_relations_per_type
    dataset.index_functions["frequency_percentiles"] = index_frequency_percentiles
    dataset.index_functions["fre"] = index_frequency_percent

    for obj in ["entity", "relation"]:
        dataset.index_functions[f"{obj}_id_to_index"] = IndexWrapper(
            _invert_ids, obj=obj
        )


@numba.njit
def where_in(x, y, not_in=False):
    """Retrieve the indices of the elements in x which are also in y.

    x and y are assumed to be 1 dimensional arrays.

    :params: not_in: if True, returns the indices of the of the elements in x
    which are not in y.

    """
    # np.isin is not supported in numba. Also: "i in y" raises an error in numba
    # setting njit(parallel=True) slows down the function
    list_y = set(y)
    return np.where(np.array([i in list_y for i in x]) != not_in)[0]
