"""Explicit population contrast definitions and validation."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

CONTRAST_TYPES = {"target_vs_all", "target_vs_one", "target_vs_set"}


@dataclass(frozen=True)
class Contrast:
    """A target population and an explicit rule for choosing alternatives."""

    target: Any
    contrast_type: str
    alternatives: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        if self.contrast_type not in CONTRAST_TYPES:
            raise ValueError(
                "`contrast_type` must be 'target_vs_all', 'target_vs_one', "
                "or 'target_vs_set'."
            )
        if self.contrast_type == "target_vs_all" and self.alternatives:
            raise ValueError("A target-versus-all contrast cannot name alternatives.")
        if self.contrast_type == "target_vs_one" and len(self.alternatives) != 1:
            raise ValueError("A target-versus-one contrast requires one alternative.")
        if self.contrast_type == "target_vs_set" and not self.alternatives:
            raise ValueError("A target-versus-set contrast requires alternatives.")
        if self.target in self.alternatives:
            raise ValueError("The target cannot also be an alternative.")
        if len(pd.Index(self.alternatives).unique()) != len(self.alternatives):
            raise ValueError("Contrast alternatives must be unique.")

    @classmethod
    def target_vs_all(cls, target: Any) -> "Contrast":
        return cls(target=target, contrast_type="target_vs_all")

    @classmethod
    def target_vs_one(cls, target: Any, alternative: Any) -> "Contrast":
        return cls(
            target=target,
            contrast_type="target_vs_one",
            alternatives=(alternative,),
        )

    @classmethod
    def target_vs_set(cls, target: Any, alternatives: Iterable[Any]) -> "Contrast":
        if isinstance(alternatives, (str, bytes)):
            raise TypeError("`alternatives` must be an iterable of group labels.")
        if isinstance(alternatives, (set, frozenset)):
            alternatives = sorted(
                alternatives, key=lambda group: (type(group).__name__, repr(group))
            )
        return cls(
            target=target,
            contrast_type="target_vs_set",
            alternatives=tuple(alternatives),
        )


@dataclass(frozen=True)
class ResolvedContrast:
    """A contrast with its complete alternative group set resolved."""

    target: Any
    contrast_type: str
    alternative_groups: tuple[Any, ...]


def _as_target_list(targets: Any) -> list[Any]:
    if isinstance(targets, (str, bytes)) or np.isscalar(targets):
        return [targets]
    if isinstance(targets, np.ndarray) and targets.ndim == 0:
        return [targets.item()]
    try:
        return list(targets)
    except TypeError:
        return [targets]


def resolve_contrasts(
    group_labels: np.ndarray,
    *,
    targets: Any = None,
    contrasts: Optional[Sequence[Contrast]] = None,
) -> list[ResolvedContrast]:
    """Validate contrasts against observed groups and resolve target-versus-all."""
    observed_groups = sorted(
        pd.unique(group_labels).tolist(),
        key=lambda group: (type(group).__name__, repr(group)),
    )
    if len(observed_groups) < 2:
        raise ValueError("Canonical profiling requires at least two observed groups.")
    if targets is not None and contrasts is not None:
        raise ValueError("Provide either `targets` or `contrasts`, not both.")

    if contrasts is None:
        requested_targets = (
            observed_groups if targets is None else _as_target_list(targets)
        )
        if not requested_targets:
            raise ValueError("`targets` cannot be empty.")
        if len(pd.Index(requested_targets).unique()) != len(requested_targets):
            raise ValueError("Requested target groups must be unique.")
        contrast_list = [Contrast.target_vs_all(target) for target in requested_targets]
    else:
        contrast_list = list(contrasts)
        if not contrast_list:
            raise ValueError("`contrasts` cannot be empty.")
        if not all(isinstance(contrast, Contrast) for contrast in contrast_list):
            raise TypeError("Every contrast must be a Contrast object.")

    resolved = []
    for contrast in contrast_list:
        if contrast.target not in observed_groups:
            raise ValueError(
                f"Target group {contrast.target!r} is not present in the data."
            )
        if contrast.contrast_type == "target_vs_all":
            alternatives = tuple(
                group for group in observed_groups if group != contrast.target
            )
        else:
            missing = [
                group for group in contrast.alternatives if group not in observed_groups
            ]
            if missing:
                raise ValueError(
                    f"Alternative groups are not present in the data: {missing}"
                )
            alternatives = contrast.alternatives
        resolved.append(
            ResolvedContrast(
                target=contrast.target,
                contrast_type=contrast.contrast_type,
                alternative_groups=alternatives,
            )
        )
    return resolved
