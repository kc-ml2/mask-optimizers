"""Mirror-paired coherent surgery on 2D binary-mask motifs."""
from __future__ import annotations

import numpy as np

from .base import BaseOptimizer, mask_key
from .fourier import FourierSpace


def components(mask: np.ndarray, value: int, periodic_x: bool) -> list[np.ndarray]:
    gx, gy = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    found = []
    for x, y in np.argwhere(mask == value):
        if seen[x, y]:
            continue
        queue = [(int(x), int(y))]
        seen[x, y] = True
        pixels = []
        while queue:
            px, py = queue.pop()
            pixels.append((px, py))
            for nx, ny in ((px-1,py),(px+1,py),(px,py-1),(px,py+1)):
                if periodic_x:
                    nx %= gx
                if 0 <= nx < gx and 0 <= ny < gy and not seen[nx,ny] and mask[nx,ny] == value:
                    seen[nx,ny] = True
                    queue.append((nx,ny))
        found.append(np.asarray(pixels, dtype=int))
    return found


def mirrored_edit(mask: np.ndarray, pixels, value: int) -> np.ndarray:
    out = mask.copy()
    gy = mask.shape[1]
    for x, y in pixels:
        out[x, y] = value
        out[x, gy-1-y] = value
    return out


class MotifSurgery(BaseOptimizer):
    def __init__(self, space, config, seed):
        if not space.mirror_y:
            raise ValueError("Mirror-paired surgery requires y-reflection symmetry")
        super().__init__(space, config, seed, "motif_surgery")
        self.radius = int(config.get("radius", 2))
        if self.radius < 1:
            raise ValueError("Edit radius must be positive")
        self.initial_count = int(config.get("initial_count", 4))
        if self.initial_count < 1:
            raise ValueError("At least one initial mask is required")
        self.incumbent = None
        self.incumbent_score = None
        self.accepted = self.rejected = 0
        self.last_operation = None
        self.geometry = FourierSpace(space, (min(2, max(1, space.shape[0]//2-1)), 1))

    def _random_mask(self):
        c = self.geometry.normalize(self.rng.normal(size=self.geometry.dimensions))
        return self.geometry.mask(c)

    def _surgery(self):
        base = self.incumbent
        gx, gy = base.shape
        for _ in range(16):
            material = int(self.rng.integers(0, 2))
            groups = components(base, material, self.space.periodic_x)
            if not groups:
                continue
            group = groups[int(self.rng.integers(len(groups)))]
            operation = ("translate", "expand", "contract", "notch_bridge")[int(self.rng.integers(4))]
            pixels = set(map(tuple, group))
            extent = min(self.radius, max(1, gx//4))
            if operation == "translate":
                shift = int(self.rng.choice([-extent, extent]))
                moved = {(int((x+shift)%gx if self.space.periodic_x else x+shift),int(y)) for x,y in pixels}
                moved = {(x,y) for x,y in moved if 0 <= x < gx}
                candidate = mirrored_edit(base, pixels, 1-material)
                candidate = mirrored_edit(candidate, moved, material)
            elif operation in {"expand", "contract"}:
                neighbors = set()
                for x,y in pixels:
                    for delta in range(-extent, extent+1):
                        nx = x+delta
                        if self.space.periodic_x:
                            nx %= gx
                        if 0 <= nx < gx:
                            neighbors.add((nx,int(y)))
                if operation == "expand":
                    candidate = mirrored_edit(base, neighbors, material)
                else:
                    boundary = neighbors - pixels
                    trimmed = {(int(x),int(y)) for x,y in pixels if any(((x+dx)%gx,y) in boundary for dx in (-1,1))}
                    candidate = mirrored_edit(base, trimmed, 1-material)
            else:
                # Paired notch or bridge at a selected motif boundary.
                x,y = map(int, group[int(self.rng.integers(len(group)))])
                line = {(((x+delta)%gx if self.space.periodic_x else x+delta),y)
                        for delta in range(-extent, extent+1)}
                line = {(xx,yy) for xx,yy in line if 0 <= xx < gx}
                candidate = mirrored_edit(base, line, 1-material if self.rng.random() < .5 else material)
            if not np.array_equal(candidate, base) and mask_key(candidate) not in self.archive:
                self.last_operation = operation
                return candidate
        self.last_operation = "restart"
        for _ in range(16):
            candidate = self._random_mask()
            if not np.array_equal(candidate,base) and mask_key(candidate) not in self.archive:
                return candidate
        raise RuntimeError("Could not produce a distinct mirrored mask after 16 restarts")

    def _ask(self):
        if self.evaluations < self.initial_count:
            self.last_operation = "initialize"
            return self._random_mask(), {"operation": "initialize"}
        return self._surgery(), {"operation": self.last_operation}

    def _tell(self, score, mask, state, key):
        if self.incumbent_score is None or score > self.incumbent_score:
            self.incumbent, self.incumbent_score = mask.copy(), score
            if state["operation"] != "initialize":
                self.accepted += 1
                self.rejected = 0
                if self.accepted % 4 == 0:
                    self.radius = min(8, self.radius+1)
        elif state["operation"] != "initialize":
            self.rejected += 1
            if self.rejected % 8 == 0:
                self.radius = max(1, self.radius-1)

    def inspect(self):
        return {**super().inspect(), "radius": self.radius, "accepted_edits": self.accepted,
                "rejected_edits": self.rejected, "last_operation": self.last_operation}
