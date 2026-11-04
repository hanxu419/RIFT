"""Declarative experiment-protocol validation for public RIFT configurations.

The model code historically supported several evaluation conventions.  Public
configs must declare which convention they use so that a causal run cannot
silently fall back to unrestricted history or filtering behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


STRICT_QUERY_TIME = "strict_query_time"
LEGACY_UNSPECIFIED = "legacy_unspecified"


class ProtocolConfigError(ValueError):
    """Raised when a declared protocol and executable options disagree."""


_MISSING = object()


def _get(config: Any, key: str, default: Any = _MISSING) -> Any:
    """Read a dotted key from a LibKGE Config or a nested mapping."""
    if hasattr(config, "get") and not isinstance(config, dict):
        try:
            return config.get(key)
        except (KeyError, ValueError):
            if default is not _MISSING:
                return default
            raise
    current = config
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            if default is not _MISSING:
                return default
            raise KeyError(key)
        current = current[part]
    return current


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def protocol_snapshot(config: Any) -> dict[str, Any]:
    """Return the protocol fields that should be saved with a run manifest."""
    keys = (
        "protocol.name",
        "protocol.task",
        "protocol.history_source_splits",
        "protocol.history_time_boundary",
        "protocol.same_time_history",
        "protocol.time_ids_chronological",
        "protocol.filter_scope",
        "protocol.validation_uses_test_labels",
        "protocol.candidate_entities",
        "protocol.ranking_directions",
        "eval.filter_splits",
        "eval.filter_time",
        "eval.filter_with_test",
        "eval.tie_handling",
        "valid.metric",
        "rift.neighbor.causal",
        "rift.neighbor.sampling",
        "rift.neighbor.order",
        "rift.neighbor.retrieval_policy",
    )
    return {key: _get(config, key, None) for key in keys}


def _expect(errors: list[str], config: Any, key: str, expected: Any) -> None:
    actual = _get(config, key, _MISSING)
    if actual is _MISSING:
        errors.append(f"missing required option {key!r}")
    elif actual != expected:
        errors.append(f"{key}={actual!r}; expected {expected!r}")


def validate_protocol_config(config: Any, *, where: str = "configuration") -> str:
    """Validate the executable options for the declared public protocol.

    ``legacy_unspecified`` remains available for low-level framework utilities. It is
    intentionally not accepted as evidence for a new public experiment.
    """
    name = str(_get(config, "protocol.name", LEGACY_UNSPECIFIED)).strip().lower()
    if name in ("", LEGACY_UNSPECIFIED):
        return LEGACY_UNSPECIFIED

    errors: list[str] = []
    _expect(errors, config, "protocol.history_source_splits", ["train"])
    _expect(errors, config, "protocol.candidate_entities", "all")
    _expect(errors, config, "protocol.ranking_directions", ["head", "tail"])
    _expect(errors, config, "eval.tie_handling", "rounded_mean_rank")
    _expect(errors, config, "train.split", "train")

    filter_splits = _as_list(_get(config, "eval.filter_splits", []))

    if name == STRICT_QUERY_TIME:
        _expect(errors, config, "protocol.history_time_boundary", "strict_lt")
        _expect(errors, config, "protocol.same_time_history", "excluded")
        _expect(errors, config, "protocol.time_ids_chronological", True)
        _expect(errors, config, "protocol.filter_scope", "query_time")
        _expect(errors, config, "protocol.validation_uses_test_labels", False)
        _expect(errors, config, "rift.neighbor.causal", True)
        _expect(errors, config, "rift.neighbor.sampling", "time_ordered")
        order = _get(config, "rift.neighbor.order", None)
        if order not in ("time_desc", "desc"):
            errors.append(
                f"rift.neighbor.order={order!r}; expected 'time_desc' or 'desc'"
            )
        _expect(errors, config, "eval.filter_time", True)
        _expect(errors, config, "eval.filter_with_test", False)
        if "test" in filter_splits:
            errors.append("strict validation must not include test in eval.filter_splits")
        if "train" not in filter_splits or "valid" not in filter_splits:
            errors.append("strict filtering must declare eval.filter_splits: [train, valid]")
        _expect(errors, config, "valid.metric", "mean_reciprocal_rank_filtered")
    else:
        errors.append(
            f"unsupported protocol.name={name!r}; expected {STRICT_QUERY_TIME!r} "
        )

    if errors:
        details = "\n  - ".join(errors)
        raise ProtocolConfigError(
            f"Protocol validation failed for {where} ({name}):\n  - {details}"
        )
    return name

