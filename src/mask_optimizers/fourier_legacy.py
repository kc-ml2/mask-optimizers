"""Independent real coordinates for the authors' symmetric Fourier level set.

Coordinate ordering and reflection follow jLabKAIST/flrl utils/utils_lsf.py,
commit 7838e71313d71cee8e2db3b432f41f80b9106a95. The grid includes both period
endpoints, as in the released code. Float64 separable evaluation avoids its
large repeated convolution grid. There are (2*Nx+1)*(Ny+1) real coordinates.
"""
from functools import lru_cache

import numpy as np


def coefficient_matrix(vector, nx, ny):
    nreal = (nx + 1) * (ny + 1)
    z = vector[:nreal] + 1j * np.pad(vector[nreal:], (0, ny + 1))
    left = z.reshape(nx + 1, ny + 1).T
    top = np.concatenate((left, left[:, :-1][:, ::-1].conj()), axis=1)
    return np.concatenate((top, top[:-1][::-1]), axis=0)


@lru_cache(maxsize=8)
def basis(nx, ny, gx, gy):
    if not (1 <= nx <= 16 and 0 <= ny <= 8 and gx >= 2 and gy >= 2):
        raise ValueError("Unsupported Fourier modes or grid")
    dimensions = (2 * nx + 1) * (ny + 1)
    if dimensions * gx * gy > 16_000_000:
        raise ValueError("Fourier basis exceeds the supported memory bound")
    matrices = np.stack([coefficient_matrix(row, nx, ny) for row in np.eye(dimensions)])
    ex = np.exp(2j * np.pi * np.outer(np.arange(-nx, nx + 1), np.linspace(0, 1, gx)))
    ey = np.exp(2j * np.pi * np.outer(np.linspace(0, 1, gy), np.arange(ny, -ny - 1, -1)))
    result = np.einsum('ym,dmn,nx->dyx', ey, matrices, ex, optimize=True).real.reshape(dimensions, -1).T.copy()
    result.flags.writeable = False
    return result


class FourierGeometry:
    def __init__(self, nx, ny, gx, gy):
        self.nx, self.ny, self.gx, self.gy = nx, ny, gx, gy
        self.basis = basis(nx, ny, gx, gy)
        self.dimensions = self.basis.shape[1]

    def field(self, coefficients):
        c = np.asarray(coefficients, dtype=float)
        if c.shape != (self.dimensions,) or not np.isfinite(c).all():
            raise ValueError("Invalid Fourier coefficients")
        return self.basis @ c

    def mask(self, coefficients):
        return (self.field(coefficients) >= 0).astype(np.uint8).tolist()

    def normalize(self, coefficients, kind="l2", fallback=None):
        c = np.asarray(coefficients, dtype=float)
        self.field(c)
        norm = np.sqrt(np.mean(self.field(c) ** 2)) if kind == "rms" else np.linalg.norm(c)
        if norm <= 1e-14:
            return np.zeros_like(c) if fallback is None else np.array(fallback, copy=True)
        return c / norm

    def gradient_request(self, coefficients, beta, material_map="index"):
        return {"coefficients": np.asarray(coefficients).tolist(), "modes_x": self.nx, "modes_y": self.ny,
                "beta": float(beta), "material_map": material_map}
