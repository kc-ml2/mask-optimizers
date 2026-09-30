"""Mask-aware low-rank covariance adaptation in a unit-sphere tangent chart."""
from __future__ import annotations

import math
import numpy as np

from .base import BaseOptimizer, mask_key
from .fourier import FourierSpace
from .h12_math import FLOOR, CAP, normalize, chart, tangent_log, covariance_refactor, weights


class TangentCovariance(BaseOptimizer):
    def __init__(self, space, config, seed):
        if not space.mirror_y:
            raise ValueError("Tangent covariance requires y-reflection symmetry")
        if space.decoder != "h12_cell_centers_v1":
            raise ValueError("H12 requires its explicit cell-centered Fourier decoder")
        super().__init__(space, config, seed, "tangent_covariance")
        self.geometry = FourierSpace(space, (int(config.get("modes_x",8)), int(config.get("modes_y",4))))
        self.d = self.geometry.dimensions
        self.k = self.d-1
        self.population = int(config.get("population",8))
        self.rank = min(int(config.get("rank",4)), self.k)
        self.initial_sigma = float(config.get("sigma",.30))
        self.adapt_covariance = config.get("adapt_covariance",True)
        self.novelty = config.get("novelty",True)
        if self.population not in {8,16,32} or not 0 <= self.rank <= self.k or not 0 < self.initial_sigma <= 1:
            raise ValueError("Invalid tangent covariance configuration")
        if type(self.adapt_covariance) is not bool or type(self.novelty) is not bool:
            raise ValueError("Covariance and novelty ablations require booleans")
        self.mean = normalize(self.rng.normal(size=self.d))
        self.diagonal = np.ones(self.k)
        self.low_rank = np.zeros((self.k,0))
        self.path = np.zeros(self.k)
        self.sigma = self.initial_sigma
        self.slot = 0
        self.generation = 0
        self.scored = []
        self.masks = {}
        self.collapse = 0
        self.last_refactor = None
        self.informative_updates = 0
        self.no_information_ties = 0
        self.restarts = 0
        self.duplicate_draws = 0

    def _draw(self, restart=False):
        if restart:
            c = normalize(self.rng.normal(size=self.d))
        else:
            v = np.sqrt(self.diagonal)*self.rng.normal(size=self.k)
            if self.low_rank.shape[1]:
                v += self.low_rank @ self.rng.normal(size=self.low_rank.shape[1])
            c = normalize(self.mean + self.sigma*(chart(self.mean) @ v)/math.sqrt(self.k))
        t = chart(self.mean).T @ tangent_log(self.mean,c) / self.sigma
        mask = self.geometry.mask(c)
        key = mask_key(mask)
        distance = min((float(np.mean(mask != old)) for old in self.masks.values()),default=1.)
        for row in self.scored:
            distance = min(distance,float(np.mean(mask != row["mask"])))
        return {"c":c,"t":t,"mask":mask,"key":key,"distance":distance}

    def _choose(self):
        reserved = self.novelty and self.slot >= self.population-max(1,round(self.population/8))
        used = {row["key"] for row in self.scored}
        for restart in (False,True):
            options = []
            for _ in range(4):
                item = self._draw(restart)
                if item["key"] in used or any(row["key"] == item["key"] for row in options):
                    self.duplicate_draws += 1
                    continue
                options.append(item)
                if not reserved:
                    break
            if options:
                return max(options,key=lambda row:row["distance"]) if reserved else options[0]
        raise RuntimeError("Covariance population collapsed to duplicate masks")

    def _ask(self):
        item = self._choose()
        self.slot += 1
        return item["mask"], item

    def _tell(self, score, mask, item, key):
        item = {**item,"score":score}
        self.scored.append(item)
        self.masks[key] = mask.copy()
        if self.slot >= self.population:
            self._finish()

    def _finish(self):
        n = len(self.scored)
        scores = [row["score"] for row in self.scored]
        if n >= 4 and len(set(scores)) >= 2:
            w = weights(scores)
            centered = w-1/n
            t = np.stack([row["t"] for row in self.scored])
            g = centered @ t
            norm = np.linalg.norm(g)
            q = chart(self.mean)
            theta = min(.5*self.sigma*norm,math.pi/3)
            new_mean = normalize(math.cos(theta)*self.mean+math.sin(theta)*(q @ g/norm)) if norm else self.mean.copy()
            cov = np.diag(self.diagonal)+self.low_rank @ self.low_rank.T
            new_path = .7*self.path+math.sqrt(.3*1.7/float(centered @ centered))*g
            new_sigma = float(np.clip(self.sigma*math.exp(.15*(np.linalg.norm(new_path)/
                                      math.sqrt(np.trace(cov)/self.k)-1)),.03,1))
            denom = 1+self.mean @ new_mean
            if denom < 1e-10:
                raise ArithmeticError("Antipodal mean transport")
            transported = q-np.outer(self.mean+new_mean,new_mean @ q/denom)
            transfer = chart(new_mean).T @ transported
            if np.linalg.norm(transfer.T @ transfer-np.eye(self.k)) > 1e-9:
                raise ArithmeticError("Nonorthogonal tangent transport")
            if self.adapt_covariance:
                star = .8*cov+.2*(FLOOR*np.eye(self.k)+(t.T*w) @ t)
                new_diag,new_low,info = covariance_refactor(transfer @ star @ transfer.T,self.rank,True)
            else:
                new_diag,new_low,info = np.ones(self.k),np.zeros((self.k,0)),None
            new_path = transfer @ new_path
            reconstructed = np.diag(new_diag)+new_low @ new_low.T
            spectrum = np.linalg.eigvalsh(reconstructed)
            if spectrum[0] < FLOOR-1e-10 or spectrum[-1] > CAP+1e-10:
                raise ArithmeticError("Covariance spectrum outside bounds")
            self.mean,self.diagonal,self.low_rank,self.path,self.sigma = new_mean,new_diag,new_low,new_path,new_sigma
            self.last_refactor = info
            self.informative_updates += 1
        elif n >= 4:
            self.no_information_ties += 1
        self.collapse = self.collapse+1 if n < 4 else 0
        if self.collapse == 3:
            self.mean = normalize(self.rng.normal(size=self.d))
            self.diagonal,self.low_rank = np.ones(self.k),np.zeros((self.k,0))
            self.path,self.sigma = np.zeros(self.k),self.initial_sigma
            self.collapse = 0
            self.restarts += 1
        self.scored = []
        self.slot = 0
        self.generation += 1

    def inspect(self):
        cov = np.diag(self.diagonal)+self.low_rank @ self.low_rank.T
        return {**super().inspect(),"generation":self.generation,"slot":self.slot,
                "mean":self.mean.tolist(),"sigma":self.sigma,"path":self.path.tolist(),
                "diagonal":self.diagonal.tolist(),"low_rank":self.low_rank.tolist(),
                "spectrum":np.linalg.eigvalsh(cov).tolist(),"informative_updates":self.informative_updates,
                "no_information_ties":self.no_information_ties,"duplicate_draws":self.duplicate_draws,
                "restarts":self.restarts,"last_refactor":self.last_refactor}
