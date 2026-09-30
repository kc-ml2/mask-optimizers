"""Tangent chart and covariance correction adapted from H12 commit b0399949537a193f1de85c2d471c62cedbec648f."""
import math
import numpy as np

FLOOR, CAP = .05, 4.

def normalize(c):
    c = np.asarray(c, dtype=np.float64)
    if not np.isfinite(c).all():
        raise ValueError("Nonfinite coefficient")
    scale = np.max(np.abs(c))
    if not scale:
        raise ValueError("Zero coefficient")
    unit = (c / scale) / np.linalg.norm(c / scale)
    if not np.isfinite(unit).all():
        raise ValueError("Invalid normalized coefficient")
    return unit


def chart(m):
    d = len(m)
    s = 1. if m[-1] >= 0 else -1.
    v = m.copy()
    v[-1] -= s
    length = np.linalg.norm(v)
    q = np.eye(d, dtype=np.float64)[:, :d-1]
    if length:
        v /= length
        q -= 2 * np.outer(v, v[:d-1])
    if np.linalg.norm(q.T @ q - np.eye(d-1)) > 1e-10 or np.linalg.norm(q.T @ m) > 1e-10:
        raise ArithmeticError("Householder frame invalid")
    return q


def tangent_log(m, c):
    h = float(np.clip(m @ c, -1, 1))
    v = c - h * m
    length = np.linalg.norm(v)
    theta = math.atan2(length, h)
    if theta >= math.pi - 1e-10:
        raise ValueError("Antipodal draw")
    return v * (theta / length) if length else np.zeros_like(m)


def canonical_eigen(matrix):
    """Canonical basis of grouped eigenspaces, largest eigenvalue first."""
    matrix = (np.asarray(matrix, dtype=np.float64) + np.asarray(matrix, dtype=np.float64).T) / 2
    if not np.isfinite(matrix).all() or matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ArithmeticError("Invalid symmetric covariance")
    values, vectors = np.linalg.eigh(matrix)
    order = sorted(range(len(values)), key=lambda i: (-values[i], i))
    values, vectors = values[order], vectors[:, order]
    tol = 1e-12 * max(1., float(np.max(np.abs(values))))
    output = []
    averaged = values.copy()
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and abs(values[end-1] - values[end]) <= tol:
            end += 1
        averaged[start:end] = float(np.mean(values[start:end]))
        projector = vectors[:, start:end] @ vectors[:, start:end].T
        selected = []
        for j in range(len(values)):
            v = projector[:, j].copy()
            for _ in range(2):
                for u in selected:
                    v -= (u @ v) * u
            norm = np.linalg.norm(v)
            if norm <= 1e-12:
                continue
            v /= norm
            nz = np.flatnonzero(np.abs(v) > 1e-12)
            if v[nz[0]] < 0:
                v = -v
            selected.append(v)
            if len(selected) == end - start:
                break
        if len(selected) != end - start:
            raise ArithmeticError("Cannot canonicalize eigenvalue group")
        output.extend(selected)
        start = end
    return averaged, np.stack(output, axis=1)


def covariance_refactor(matrix, rank=4, details=False):
    """Floor, capped eigenspectrum, diagonal residual, then cap ACTUAL Cr."""
    c = np.asarray(matrix, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or not np.isfinite(c).all():
        raise ValueError("Invalid covariance")
    k = len(c)
    if not 0 <= rank <= k:
        raise ValueError("Invalid rank")
    c = (c + c.T) / 2
    pre = np.linalg.eigvalsh(c)
    if pre[0] < -1e-10:
        raise ArithmeticError("Covariance is not PSD")
    eig, vec = canonical_eigen(c)
    clipped = np.clip(eig, FLOOR, CAP)
    target = (vec * clipped) @ vec.T
    u = vec[:, :rank] * np.sqrt(np.maximum(clipped[:rank] - FLOOR, 0))
    residual = target - u @ u.T
    a = np.clip(np.diag(residual), FLOOR, CAP)
    actual = (np.diag(a) + u @ u.T)
    actual = (actual + actual.T) / 2
    spectrum = np.linalg.eigvalsh(actual)
    if not np.isfinite(spectrum).all() or spectrum[0] < FLOOR - 1e-10:
        raise ArithmeticError("Reconstruction violates covariance floor")
    top = float(spectrum[-1])
    shrink = (CAP - FLOOR) / (top - FLOOR) if top > CAP else 1.
    a = FLOOR + shrink * (a - FLOOR)
    u = math.sqrt(shrink) * u
    final = np.diag(a) + u @ u.T
    post = np.linalg.eigvalsh((final + final.T) / 2)
    if post[0] < FLOOR - 1e-10 or post[-1] > CAP + 1e-10:
        raise ArithmeticError("Actual reconstructed covariance exceeds bounds")
    info = {"pre_spectrum": pre.tolist(), "clipped_spectrum": clipped.tolist(),
            "unshrunk_max": top, "shrink": shrink, "post_min": float(post[0]),
            "post_max": float(post[-1]), "approximation_error": float(np.linalg.norm(final-target))}
    return (a, u, info) if details else (a, u)


def weights(scores):
    n = len(scores)
    mu = n // 2
    b = np.maximum(0, np.log(mu + .5) - np.log(np.arange(1, n+1)))
    b /= b.sum()
    order = sorted(range(n), key=lambda i: -scores[i])
    w = np.empty(n)
    for score in set(scores):
        positions = [j for j, i in enumerate(order) if scores[i] == score]
        for j in positions:
            w[order[j]] = np.mean(b[positions])
    return w
