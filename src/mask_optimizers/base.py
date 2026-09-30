"""Shared mechanics for solver-independent, checkpointable mask optimizers."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import numpy as np

from .contracts import MaskSpace, Observation, Proposal, content_hash


def mask_key(mask: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(mask, dtype=np.uint8).tobytes()).hexdigest()


class BaseOptimizer:
    def __init__(self, space: MaskSpace, config: dict, seed: int, name: str):
        self.space, self.config, self.name = space, dict(config), name
        self.identity = content_hash([name, space.__dict__, config, seed])
        self.rng = np.random.Generator(np.random.PCG64DXSM(seed))
        self.pending = None
        self.pending_state = None
        self.sequence = 0
        self.evaluations = 0
        self.best_score = None
        self.best_mask = None
        self.archive = {}

    def propose(self, max_candidates=1) -> list[Proposal]:
        if max_candidates != 1 or self.pending is not None:
            raise ValueError("One outstanding proposal is supported")
        mask, state = self._ask()
        value = self.space.validate(mask)
        self.sequence += 1
        self.pending = Proposal(f"{self.name}_{self.sequence}", value.reshape(-1).tolist())
        self.pending_state = state
        return [self.pending]

    def observe(self, observations: list[Observation]) -> None:
        if len(observations) != 1 or self.pending is None:
            raise ValueError("Expected exactly one pending observation")
        obs = observations[0]
        if obs.proposal_id != self.pending.id or list(obs.candidate) != self.pending.candidate:
            raise ValueError("Observation differs from the outstanding proposal")
        mask = self.space.validate(obs.candidate)
        key = mask_key(mask)
        previous = self.archive.get(key)
        if previous is not None and not np.isclose(previous, obs.score, rtol=0, atol=1e-10):
            raise ValueError("Repeated mask has inconsistent objective value")
        self.archive[key] = obs.score
        if self.best_score is None or obs.score > self.best_score:
            self.best_score, self.best_mask = float(obs.score), mask.copy()
        self._tell(float(obs.score), mask, self.pending_state, key)
        self.evaluations += 1
        self.pending = self.pending_state = None

    def checkpoint(self):
        return {"identity": self.identity, "state": deepcopy({k:v for k,v in self.__dict__.items()
                if k not in {"space", "geometry", "geometries"}})}

    def restore(self, manifest):
        if manifest["identity"] != self.identity:
            raise ValueError("Checkpoint belongs to a different optimizer configuration")
        self.__dict__.update(deepcopy(manifest["state"]))

    def inspect(self):
        return {"method": self.name, "evaluations": self.evaluations, "best_score": self.best_score,
                "unique_masks": len(self.archive), "pending": self.pending is not None}

    def export_artifacts(self):
        return []

    def close(self):
        pass
