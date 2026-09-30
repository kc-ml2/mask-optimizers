"""Standalone optimizers for structured binary masks.

All public candidates are flat x-major masks (``shape=(x_cells, y_cells)``).
Scores are finite scalar utilities to maximize; minimization callers negate them.
"""
from __future__ import annotations

from types import SimpleNamespace

from .contracts import MaskSpace, Observation, Proposal, RelaxedEvaluation, Runner, content_hash
from .fourier import FourierSpace
from .motif import MotifSurgery
from .nested import NestedFourier
from .phenotype import PhenotypeDE
from .tangent_covariance import TangentCovariance

__version__ = "0.1.0"

LEGACY_METHODS = frozenset({"flrl_lsf_random", "flrl_lsf_es", "flrl_ppo",
                             "flrl_autograd_adam", "flrl_ppo_polish", "flrl_ppo_residual"})
METHODS = LEGACY_METHODS | frozenset({"motif_surgery", "nested_fourier",
                                     "tangent_covariance", "phenotype_de"})


def create_optimizer(name: str, space: MaskSpace, config: dict | None = None, seed: int = 0):
    """Construct one independent algorithm; no evaluator or campaign is imported."""
    if name not in METHODS:
        raise ValueError(f"Unknown optimizer: {name}")
    config = dict(config or {})
    if name in LEGACY_METHODS:
        if not space.mirror_y or space.decoder != "flrl_endpoints_v1":
            raise ValueError("Ported FLRL methods require mirrored FLRL endpoint geometry")
        from .legacy import FourierOptimizer
        instance = SimpleNamespace(space=space,configuration={"grid_x":space.shape[0],"grid_y":space.shape[1]},
                                   evaluation_identity=content_hash(space.__dict__))
        return FourierOptimizer(name,instance,config,seed)
    return {"motif_surgery":MotifSurgery, "nested_fourier":NestedFourier,
            "tangent_covariance":TangentCovariance,"phenotype_de":PhenotypeDE}[name](space,config,seed)


__all__ = ["MaskSpace","Observation","Proposal","RelaxedEvaluation","Runner","FourierSpace",
           "METHODS","create_optimizer"]
