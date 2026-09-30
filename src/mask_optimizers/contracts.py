"""Small, solver-independent ask/tell contract for mask search."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping
import hashlib
import json
import math
import numpy as np


def content_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class MaskSpace:
    shape: tuple[int, int]
    mirror_y: bool = True
    periodic_x: bool = True
    decoder: str = "flrl_endpoints_v1"

    def __post_init__(self):
        if len(self.shape) != 2 or min(self.shape) < 2 or not all(isinstance(n, int) for n in self.shape):
            raise ValueError("Mask shape must contain two positive integer axes")
        if self.decoder not in {"flrl_endpoints_v1", "h12_cell_centers_v1"}:
            raise ValueError("Unknown Fourier decoder convention")

    @property
    def cells(self) -> int:
        return self.shape[0] * self.shape[1]

    def validate(self, mask) -> np.ndarray:
        value = np.asarray(mask)
        if value.shape == (self.cells,):
            value = value.reshape(self.shape)
        if value.shape != self.shape or not np.all((value == 0) | (value == 1)):
            raise ValueError("Candidate is not a binary mask of the declared shape")
        result = value.astype(np.uint8, copy=True)
        if self.mirror_y and not np.array_equal(result, result[:, ::-1]):
            raise ValueError("Candidate violates y-reflection symmetry")
        return result


@dataclass(frozen=True)
class Proposal:
    id: str
    candidate: list[int]
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Observation:
    proposal_id: str
    candidate: list[int]
    score: float
    metadata: dict = field(default_factory=dict)
    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if not math.isfinite(self.score) or any(not math.isfinite(float(v)) for v in self.metrics.values()):
            raise ValueError("Observation must have finite scores")


@dataclass(frozen=True)
class RelaxedEvaluation:
    value: float
    gradient: np.ndarray
    metrics: Mapping[str, float] = field(default_factory=dict)


class Runner:
    """Inject trusted callbacks; optimizers never own the numerical evaluator."""
    def __init__(self, optimizer, score: Callable[[np.ndarray], float],
                 gradient: Callable[[np.ndarray, float, str], RelaxedEvaluation] | None = None):
        self.optimizer, self.score, self.gradient = optimizer, score, gradient

    def step(self) -> tuple[Proposal, Observation]:
        proposal = self.optimizer.propose()[0]
        mask = self.optimizer.space.validate(proposal.candidate)
        value = float(self.score(mask))
        metadata = {}
        request = proposal.metadata.get("flrl_gradient")
        if request is not None:
            if self.gradient is None:
                raise ValueError("This optimizer requires a relaxed-gradient callback")
            coeff = np.asarray(request["coefficients"], dtype=float)
            relaxed = self.gradient(coeff, float(request["beta"]), request["material_map"])
            if not isinstance(relaxed, RelaxedEvaluation):
                raise ValueError("Gradient callback must return RelaxedEvaluation")
            grad = np.asarray(relaxed.gradient, dtype=float)
            if grad.shape != coeff.shape or not np.isfinite(grad).all() or not math.isfinite(relaxed.value):
                raise ValueError("Gradient callback returned an invalid relaxed value or vector")
            if any(not math.isfinite(float(v)) for v in relaxed.metrics.values()):
                raise ValueError("Gradient callback returned nonfinite relaxed metrics")
            metadata["flrl_gradient"] = {"gradient": grad.tolist(),
                                           "te": relaxed.metrics.get("te", relaxed.value),
                                           "tm": relaxed.metrics.get("tm", relaxed.value)}
        observation = Observation(proposal.id, proposal.candidate, value, metadata)
        self.optimizer.observe([observation])
        return proposal, observation
