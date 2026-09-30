# mask-optimizers

An independently installable Python library for optimizers of structured 2D
binary masks. It contains no simulator or campaign service. The caller owns
evaluation and supplies a finite scalar **utility to maximize**; negate a
minimization objective. Gradient methods additionally require a callback that
returns the derivative of a relaxed utility with respect to Fourier
coefficients and its relaxed utility value. Hard-mask scores remain separate
from relaxed scores and gradients.

Install from this directory with `uv pip install .`; add the `learners` extra
for PPO methods. NumPy is the only core runtime dependency.

```python
import numpy as np
from mask_optimizers import MaskSpace, Runner, create_optimizer

space = MaskSpace(shape=(16, 12), mirror_y=True, periodic_x=True)
optimizer = create_optimizer("nested_fourier", space, seed=7)
runner = Runner(optimizer, score=lambda mask: float(np.mean(mask[:, :3])))
for _ in range(20):
    proposal, observation = runner.step()
print(optimizer.inspect())
saved = optimizer.checkpoint()
resumed = create_optimizer("nested_fourier", space, seed=7)
resumed.restore(saved)
```

For gradient methods, pass `Runner(optimizer, hard_score, relaxed_gradient)`.
The latter receives `(coefficients, beta, material_map)` and returns
`RelaxedEvaluation(value, gradient, metrics={"te": ..., "tm": ...})`.
If no per-polarization metrics exist, omit `metrics`; both auxiliary inputs
default to the relaxed scalar value. A missing or nonfinite gradient is an
error. The callback must differentiate the same relaxed value it reports.

`shape` is `(x_cells, y_cells)`. Every public proposal contains a flat x-major
binary mask. `ask`/`tell` are exposed as `propose`/`observe` with one outstanding
proposal; `Runner` is an optional callback adapter. Repeatable observations
must have the same score for an identical mask. An optimizer checkpoint is a
Python object for trusted local persistence; do not unpickle untrusted files.

The two Fourier profiles are explicit and distinct. `flrl_endpoints_v1` uses
the released FLRL endpoint-inclusive grid and real-then-imaginary coordinate
order. `h12_cell_centers_v1` uses cell-centered samples and y-mode-major
cosine/sine coordinates. The H12 covariance method requires the latter profile;
the six ported FLRL methods require the former. Nested Fourier continuation and
phenotype DE accept either profile when declared at construction.

| Method | Mechanism | Additional requirement |
| --- | --- | --- |
| `flrl_lsf_random` | Independent normalized Fourier coefficients | FLRL decoder |
| `flrl_lsf_es` | Elite-recombined Gaussian search | FLRL decoder |
| `flrl_ppo` | PPO coefficient increments | FLRL decoder, Torch |
| `flrl_autograd_adam` | Adam on a relaxed coefficient objective | FLRL decoder, gradient callback |
| `flrl_ppo_polish` | Raw PPO action then exactly two local Adam steps, with hard-score rollback | FLRL decoder, Torch, gradient callback |
| `flrl_ppo_residual` | Tangent Adam move plus Gaussian PPO residual | FLRL decoder, Torch, gradient callback unless gradient radius is zero |
| `motif_surgery` | Connected-component translation, expansion, contraction, and paired notch/bridge edits | Mirrored masks |
| `nested_fourier` | Paired boundary perturbations and exact incumbent-preserving mode lifts | Fourier decoder |
| `tangent_covariance` | H12 rank-selected tangent covariance and novelty sampling | Cell-centered decoder |
| `phenotype_de` | Differential vectors from fill/variation descriptor bins | Fourier decoder |

Methods: `flrl_lsf_random`, `flrl_lsf_es`, `flrl_ppo`,
`flrl_autograd_adam`, `flrl_ppo_polish`, `flrl_ppo_residual`,
`motif_surgery`, `nested_fourier`, `tangent_covariance`, and `phenotype_de`.
The PPO methods require `pip install mask-optimizers[learners]`. The two hybrids
are ports of existing dqn-meent implementations. The original review-rejected
residual draft is superseded by `flrl_ppo_residual` and is not a separate
algorithm.

The four new methods use documented starting parameters, not tuned claims:
motif surgery uses four initial masks and radius two; nested continuation uses
stages `(2,1),(4,2),(6,3),(8,4)` with 64 unsuccessful pairs before lifting;
tangent covariance uses population eight, rank four, and step size 0.30;
phenotype DE uses population 16, 10×10 descriptor bins, differential weight
0.5, and mode-group crossover probability 0.5. These are configurable through
the `config` mapping. Library completion depends on contract, mechanism, and
replay tests, not efficiency on a particular simulator.

The six FLRL ports preserve synthetic proposal/update traces from dqn-meent
commit `0d9183aba76fa36eb72f203318c64efdfac6bda8`. The H12 decoder,
tangent update, and covariance correction preserve frozen synthetic traces from
submitted implementation commit `b0399949537a193f1de85c2d471c62cedbec648f`.
H12's fixed physics identity and solver accounting were left in its original
service package; this library version uses only its numerical algorithm.
Tests check numerical invariants, source parity, 256×128 mask construction,
and deterministic continuation. No MEENT result or problem-specific
effectiveness claim is part of this package.
