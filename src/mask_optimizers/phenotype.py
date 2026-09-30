"""Differential evolution with an archive indexed by physical mask descriptors."""
from __future__ import annotations

import numpy as np

from .base import BaseOptimizer, mask_key
from .fourier import FourierSpace


def descriptor(mask: np.ndarray, bins: int) -> tuple[int, int]:
    fill = float(np.mean(mask))
    variation = float(np.mean(mask[:, 1:] != mask[:, :-1]))
    return min(bins-1, int(fill*bins)), min(bins-1, int(variation*bins))


class PhenotypeDE(BaseOptimizer):
    def __init__(self, space, config, seed):
        if not space.mirror_y:
            raise ValueError("Phenotype archive requires y-reflection symmetry")
        super().__init__(space, config, seed, "phenotype_de")
        self.geometry = FourierSpace(space, (int(config.get("modes_x", 8)), int(config.get("modes_y", 4))))
        self.population_size = int(config.get("population_size", 16))
        self.bins = int(config.get("bins", 10))
        self.differential_weight = float(config.get("differential_weight", .5))
        self.crossover_rate = float(config.get("crossover_rate", .5))
        if self.population_size < 4 or self.bins < 2 or not 0 < self.differential_weight <= 2 or not 0 < self.crossover_rate <= 1:
            raise ValueError("Invalid differential-evolution configuration")
        self.representatives = {}
        self.phenotypes = {}
        self.bin_visits = {}
        self.duplicate_draws = 0
        self.restarts = 0
        self.best_coefficients = None

    def _random(self):
        return self.geometry.normalize(self.rng.normal(size=self.geometry.dimensions))

    def _groups(self):
        nx,ny = self.geometry.modes
        if self.space.decoder == "h12_cell_centers_v1":
            return [np.arange(n*(2*nx+1),(n+1)*(2*nx+1)) for n in range(ny+1)]
        real = (nx+1)*(ny+1)
        return [np.array([p*(ny+1)+n for p in range(nx+1)] +
                         [real+p*(ny+1)+n for p in range(nx)]) for n in range(ny+1)]

    def _offspring(self):
        rows = list(self.representatives.values())
        if len(rows) < 3:
            return self._random()
        if self.rng.random() < .5:
            ranked = sorted(rows, key=lambda row: row["score"], reverse=True)
            pool = ranked[:max(3, len(ranked)//2)]
            target = pool[int(self.rng.integers(len(pool)))]
        else:
            sparse = sorted(rows, key=lambda row: (self.bin_visits.get(row["bin"],0), -row["score"]))
            target = sparse[int(self.rng.integers(min(3,len(sparse))))]
        donors = [row for row in rows if row["key"] != target["key"]]
        if len(donors) < 2:
            return self._random()
        first,second = self.rng.choice(len(donors),size=2,replace=False)
        mutant = target["c"] + self.differential_weight*(donors[first]["c"]-donors[second]["c"])
        trial = target["c"].copy()
        groups = self._groups()
        forced = int(self.rng.integers(len(groups)))
        for i,group in enumerate(groups):
            if i == forced or self.rng.random() < self.crossover_rate:
                trial[group] = mutant[group]
        return self.geometry.normalize(trial, fallback=target["c"])

    def _ask(self):
        for attempt in range(16):
            c = self._random() if self.evaluations < self.population_size or attempt >= 8 else self._offspring()
            mask = self.geometry.mask(c)
            if mask_key(mask) not in self.archive:
                if attempt >= 8:
                    self.restarts += 1
                return mask, {"coefficients": c}
            self.duplicate_draws += 1
        raise RuntimeError("Could not generate a distinct phenotype after 16 retries")

    def _tell(self, score, mask, state, key):
        c = state["coefficients"].copy()
        self.representatives[key] = {"key": key, "c": c, "score": score,
                                     "bin": descriptor(mask, self.bins)}
        bin_id = descriptor(mask, self.bins)
        self.bin_visits[bin_id] = self.bin_visits.get(bin_id,0)+1
        previous = self.phenotypes.get(bin_id)
        if previous is None or score > previous["score"]:
            self.phenotypes[bin_id] = self.representatives[key]
        if self.best_mask is not None and np.array_equal(mask,self.best_mask):
            self.best_coefficients = c
        # Donor pool consists of one representative per bin plus the global best.
        keep = {row["key"] for row in self.phenotypes.values()}
        if self.best_mask is not None:
            keep.add(mask_key(self.best_mask))
        self.representatives = {k:v for k,v in self.representatives.items() if k in keep}

    def inspect(self):
        return {**super().inspect(), "occupied_bins": len(self.phenotypes),
                "duplicate_draws": self.duplicate_draws, "restarts": self.restarts}
