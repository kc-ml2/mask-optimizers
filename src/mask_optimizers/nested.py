"""Incumbent-preserving nested Fourier-band boundary continuation."""
from __future__ import annotations

import numpy as np

from .base import BaseOptimizer
from .fourier import FourierSpace


class NestedFourier(BaseOptimizer):
    def __init__(self, space, config, seed):
        if not space.mirror_y:
            raise ValueError("Nested Fourier bands require y reflection")
        super().__init__(space, config, seed, "nested_fourier")
        modes = config.get("modes", ((2,1),(4,2),(6,3),(8,4)))
        self.modes = tuple(tuple(map(int, pair)) for pair in modes)
        if not self.modes or any(b[0] < a[0] or b[1] < a[1] for a,b in zip(self.modes,self.modes[1:])):
            raise ValueError("Fourier modes must be nonempty and nested")
        self.geometries = tuple(FourierSpace(space, mode) for mode in self.modes)
        self.stage = 0
        self.coefficients = self.geometries[0].normalize(self.rng.normal(size=self.geometries[0].dimensions))
        self.incumbent_score = None
        self.pair_index = 0
        self.pair = None
        self.stagnation = 0
        self.stage_patience = int(config.get("stage_patience", 64))
        if self.stage_patience < 1:
            raise ValueError("stage_patience must be positive")

    @property
    def geometry(self):
        return self.geometries[self.stage]

    def _lift(self):
        if self.stage == len(self.geometries)-1:
            return
        before = self.geometry.mask(self.coefficients)
        lifted = self.geometry.lift(self.coefficients, self.geometries[self.stage+1])
        self.stage += 1
        if not np.array_equal(before, self.geometry.mask(lifted)):
            raise ArithmeticError("Nested Fourier lift changed the incumbent mask")
        self.coefficients = lifted
        self.stagnation = 0

    def _new_pair(self):
        geom = self.geometry
        for _ in range(16):
            direction = self.rng.normal(size=geom.dimensions)
            if self.stage and self.pair_index % 8 != 7:
                # Select just newly admitted harmonics via a lifted old-band basis.
                old = self.geometries[self.stage-1]
                old_support = old.lift(np.ones(old.dimensions), geom) != 0
                direction[old_support] = 0
            norm = np.linalg.norm(direction)
            if norm < 1e-12:
                continue
            direction /= norm
            base = geom.mask(self.coefficients)
            amplitude = .01
            target = None
            for _ in range(12):
                trial = geom.normalize(self.coefficients + amplitude*direction)
                changed = np.mean(geom.mask(trial) != base)
                if .005 <= changed <= .05:
                    target = amplitude
                    break
                amplitude *= 2 if changed < .005 else .5
            if target is None:
                target = amplitude
            plus = geom.normalize(self.coefficients + target*direction)
            minus = geom.normalize(self.coefficients - target*direction)
            if np.array_equal(geom.mask(plus), base) and np.array_equal(geom.mask(minus), base):
                continue
            self.pair_index += 1
            return {"candidates": (plus, minus), "scores": [], "index": 0,
                    "old_band": self.stage == 0 or self.pair_index % 8 == 0}
        # A new seeded direction is a bounded fallback when all small moves vanish.
        trial = geom.normalize(self.rng.normal(size=geom.dimensions))
        self.pair_index += 1
        return {"candidates": (trial, self.coefficients.copy()), "scores": [], "index": 0, "old_band": True}

    def _ask(self):
        if self.incumbent_score is None:
            return self.geometry.mask(self.coefficients), {"initial": True, "coefficients": self.coefficients.copy()}
        if self.pair is None:
            self.pair = self._new_pair()
        i = self.pair["index"]
        c = self.pair["candidates"][i]
        return self.geometry.mask(c), {"initial": False, "coefficients": c.copy(), "pair_index": i}

    def _tell(self, score, mask, state, key):
        if state["initial"]:
            self.incumbent_score = score
            return
        self.pair["scores"].append(score)
        self.pair["index"] += 1
        if self.pair["index"] < 2:
            return
        best = int(np.argmax(self.pair["scores"]))
        if self.pair["scores"][best] > self.incumbent_score:
            self.coefficients = self.pair["candidates"][best].copy()
            self.incumbent_score = self.pair["scores"][best]
            self.stagnation = 0
        else:
            self.stagnation += 1
        self.pair = None
        if self.stagnation >= self.stage_patience:
            self._lift()

    def inspect(self):
        return {**super().inspect(), "stage": self.stage, "modes": self.modes[self.stage],
                "stagnation_pairs": self.stagnation, "incumbent_score": self.incumbent_score}
