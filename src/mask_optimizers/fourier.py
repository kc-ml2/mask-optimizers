"""Explicitly versioned Fourier decoders; masks use flat x-major order."""
from __future__ import annotations

from functools import lru_cache
import numpy as np

from .contracts import MaskSpace
from .fourier_legacy import FourierGeometry


@lru_cache(maxsize=24)
def _center_basis(nx: int, ny: int, gx: int, gy: int) -> np.ndarray:
    if nx < 1 or ny < 0 or gx < 2 or gy < 2 or (2*nx+1)*(ny+1)*gx*gy > 16_000_000:
        raise ValueError("Unsupported Fourier basis size")
    x = (np.arange(gx, dtype=np.float64) + .5) / gx
    y = (np.arange(gy, dtype=np.float64) + .5) / gy
    columns = []
    for n in range(ny + 1):
        cy = np.cos(2*np.pi*n*y)[None, :]
        columns.append(np.broadcast_to(cy, (gx, gy)))
        for p in range(1, nx + 1):
            columns.append(np.cos(2*np.pi*p*x)[:, None] * cy)
            columns.append(np.sin(2*np.pi*p*x)[:, None] * cy)
    result = np.stack(columns, axis=-1).reshape(gx*gy, -1).copy()
    result.flags.writeable = False
    return result


class FourierSpace:
    def __init__(self, space: MaskSpace, modes: tuple[int, int]):
        self.space, self.modes = space, tuple(map(int, modes))
        self.nx, self.ny = self.modes
        self.dimensions = (2*self.nx + 1)*(self.ny + 1)
        if space.decoder == "h12_cell_centers_v1":
            self.basis = _center_basis(self.nx, self.ny, *space.shape)
        else:
            legacy = FourierGeometry(self.nx, self.ny, *space.shape)
            self.basis = legacy.basis.reshape(space.shape[1], space.shape[0], -1).transpose(1, 0, 2).reshape(space.cells, -1)

    def field(self, coefficients) -> np.ndarray:
        c = np.asarray(coefficients, dtype=np.float64)
        if c.shape != (self.dimensions,) or not np.isfinite(c).all():
            raise ValueError("Invalid Fourier coefficients")
        return (self.basis @ c).reshape(self.space.shape)

    def mask(self, coefficients) -> np.ndarray:
        return (self.field(coefficients) >= 0).astype(np.uint8)

    def normalize(self, coefficients, *, fallback=None) -> np.ndarray:
        c = np.asarray(coefficients, dtype=np.float64)
        if c.shape != (self.dimensions,) or not np.isfinite(c).all():
            raise ValueError("Invalid Fourier coefficients")
        scale = float(np.max(np.abs(c)))
        if scale == 0:
            if fallback is None:
                raise ValueError("Zero coefficient vector")
            return np.asarray(fallback, dtype=np.float64).copy()
        c = c / scale
        return c / np.linalg.norm(c)

    def lift(self, coefficients, target: 'FourierSpace') -> np.ndarray:
        if self.space != target.space or target.nx < self.nx or target.ny < self.ny:
            raise ValueError("Fourier lift requires nested modes and identical space")
        source = np.asarray(coefficients, dtype=np.float64)
        if source.shape != (self.dimensions,):
            raise ValueError("Invalid source coefficient dimension")
        out = np.zeros(target.dimensions)
        if self.space.decoder == "h12_cell_centers_v1":
            for n in range(self.ny + 1):
                src = n*(2*self.nx + 1)
                dst = n*(2*target.nx + 1)
                out[dst:dst+2*self.nx+1] = source[src:src+2*self.nx+1]
        else:
            old_real, new_real = (self.nx+1)*(self.ny+1), (target.nx+1)*(target.ny+1)
            offset_x, offset_y = target.nx-self.nx, target.ny-self.ny
            for p in range(self.nx + 1):
                for n in range(self.ny + 1):
                    out[(p+offset_x)*(target.ny+1)+n+offset_y] = source[p*(self.ny+1)+n]
            for p in range(self.nx):
                for n in range(self.ny + 1):
                    out[new_real+(p+offset_x)*(target.ny+1)+n+offset_y] = source[old_real+p*(self.ny+1)+n]
        return out
