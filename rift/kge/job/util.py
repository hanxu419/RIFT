import torch
from torch import Tensor


def get_sp_po_coords_from_spo_batch(
    batch: Tensor, num_entities: int, sp_index: dict, po_index: dict,
    temporal: bool = False,
) -> torch.Tensor:
    """Look up known head and tail answers for a batch of quadruples.

    With temporal=True, index keys are (s,p,t) and (p,o,t); otherwise the
    legacy keys (s,p) and (p,o) are used. Returns coordinates of a two-way
    binary label tensor with 2*num_entities columns.

    """
    num_ones = 0
    NOTHING = torch.zeros([0], dtype=torch.long)
    for i, triple in enumerate(batch):
        s, p, o = triple[0].item(), triple[1].item(), triple[2].item()
        t = triple[3].item() if temporal else None
        sp_key = (s, p, t) if temporal else (s, p)
        po_key = (p, o, t) if temporal else (p, o)
        num_ones += len(sp_index.get(sp_key, NOTHING))
        num_ones += len(po_index.get(po_key, NOTHING))

    coords = torch.zeros([num_ones, 2], dtype=torch.long)
    current_index = 0
    for i, triple in enumerate(batch):
        s, p, o = triple[0].item(), triple[1].item(), triple[2].item()
        t = triple[3].item() if temporal else None
        sp_key = (s, p, t) if temporal else (s, p)
        po_key = (p, o, t) if temporal else (p, o)

        objects = sp_index.get(sp_key, NOTHING)
        coords[current_index : (current_index + len(objects)), 0] = i
        coords[current_index : (current_index + len(objects)), 1] = objects
        current_index += len(objects)

        subjects = po_index.get(po_key, NOTHING) + num_entities
        coords[current_index : (current_index + len(subjects)), 0] = i
        coords[current_index : (current_index + len(subjects)), 1] = subjects
        current_index += len(subjects)

    return coords


def coord_to_sparse_tensor(
    nrows: int, ncols: int, coords: Tensor, device: str, value=1.0
):
    return torch.sparse_coo_tensor(
        coords.long().t().to(device),
        torch.full((len(coords),), value, dtype=torch.float, device=device),
        (nrows, ncols),
        device=device,
        check_invariants=True,
    )
