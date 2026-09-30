"""Numerical and lifecycle checks independent of a particular simulator."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from mask_optimizers import (METHODS, FourierSpace, MaskSpace, Observation, RelaxedEvaluation, Runner,
                             create_optimizer)
from mask_optimizers.h12_math import covariance_refactor
from mask_optimizers.motif import components, mirrored_edit
from mask_optimizers.phenotype import descriptor


@pytest.fixture(autouse=True)
def single_thread():
    try:
        import torch
    except ImportError:
        yield
        return
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


def space_for(name, shape=(16,12)):
    return MaskSpace(shape, decoder="h12_cell_centers_v1" if name == "tangent_covariance" else "flrl_endpoints_v1")


def settings(name):
    return {"level_set_modes_x":2,"level_set_modes_y":1,"modes_x":2,"modes_y":1,
            "modes":((2,1),(3,1)),"stage_patience":1,"population":8,
            "population_size":4,"initial_count":2,"episode_length":2,"rollout_steps":2}


WEIGHTS = np.random.default_rng(82).normal(size=(16,12))


def score(mask):
    return float(np.mean(mask*WEIGHTS))


def gradient(c,beta,material_map):
    return RelaxedEvaluation(-float(np.dot(c,c)),-2*np.asarray(c))


def run_steps(opt, count):
    runner = Runner(opt,score,gradient)
    return [(p.candidate,o.score) for p,o in (runner.step() for _ in range(count))]


@pytest.mark.parametrize("name",sorted(METHODS))
def test_seeded_synthetic_runs_and_checkpoint_replay(name):
    if name in {"flrl_ppo","flrl_ppo_polish","flrl_ppo_residual"}:
        pytest.importorskip("torch")
    config = settings(name)
    space = space_for(name)
    for seed in (5,6,7):
        original = create_optimizer(name,space,config,seed)
        run_steps(original,8)
        saved = original.checkpoint()
        expected = run_steps(original,8)
        resumed = create_optimizer(name,space,config,seed)
        resumed.restore(saved)
        assert run_steps(resumed,8) == expected
        assert resumed.inspect() == original.inspect()
        assert original.best_score is not None
        if name in {"flrl_ppo","flrl_ppo_polish","flrl_ppo_residual"}:
            assert original.ppo.updates > 0


@pytest.mark.parametrize("name",sorted(METHODS))
def test_pending_proposal_survives_checkpoint(name):
    if name in {"flrl_ppo","flrl_ppo_polish","flrl_ppo_residual"}:
        pytest.importorskip("torch")
    opt = create_optimizer(name,space_for(name),settings(name),seed=17)
    proposal = opt.propose()[0]
    resumed = create_optimizer(name,space_for(name),settings(name),seed=17)
    resumed.restore(opt.checkpoint())
    assert resumed.pending == proposal
    observed = Observation(proposal.id,proposal.candidate,score(np.array(proposal.candidate).reshape(16,12)),
                           {"flrl_gradient":{"gradient":(-2*np.array(proposal.metadata["flrl_gradient"]["coefficients"])).tolist(),
                                             "te":0.,"tm":0.}} if "flrl_gradient" in proposal.metadata else {})
    opt.observe([observed])
    resumed.observe([observed])
    assert opt.propose()[0] == resumed.propose()[0]


def test_legacy_ports_match_frozen_source_traces():
    pytest.importorskip("torch")
    data = json.loads((Path(__file__).parent/"fixtures/legacy_traces.json").read_text())
    for name,rows in data["traces"].items():
        opt = create_optimizer(name,MaskSpace((16,12)),data["config"],seed=data["seed"])
        for row in rows:
            proposal = opt.propose()[0]
            mask = np.array(proposal.candidate,dtype=np.uint8).reshape(16,12)
            assert hashlib.sha256(mask.tobytes()).hexdigest() == row["mask_sha256"]
            value = score(mask)
            metadata = {}
            if "flrl_gradient" in proposal.metadata:
                c = np.asarray(proposal.metadata["flrl_gradient"]["coefficients"])
                metadata["flrl_gradient"] = {"gradient":(-2*c).tolist(),"te":value,"tm":value}
            opt.observe([Observation(proposal.id,proposal.candidate,value,metadata)])
            assert opt.decisions == row["decisions"]
            assert (opt.ppo.updates if opt.ppo else opt.adam_step) == row["updates"]


@pytest.mark.parametrize("decoder",("flrl_endpoints_v1","h12_cell_centers_v1"))
def test_fourier_lift_preserves_field_and_mask(decoder):
    space = MaskSpace((16,12),decoder=decoder)
    for old,new in (((2,1),(4,2)),((4,2),(6,3)),((6,3),(8,4))):
        source,target = FourierSpace(space,old),FourierSpace(space,new)
        c = np.random.default_rng(11).normal(size=source.dimensions)
        lifted = source.lift(c,target)
        np.testing.assert_allclose(source.field(c),target.field(lifted),atol=1e-12)
        np.testing.assert_array_equal(source.mask(c),target.mask(lifted))


def test_centered_decoder_matches_independent_full_grid_formula():
    space = MaskSpace((256,128),decoder="h12_cell_centers_v1")
    for nx,ny in ((2,1),(8,4)):
        geometry = FourierSpace(space,(nx,ny))
        c = np.zeros(geometry.dimensions)
        c[0],c[1],c[2],c[2*nx+1] = .173,.41,-.29,.23
        x = (np.arange(256)+.5)/256
        y = (np.arange(128)+.5)/128
        level = .173 + .41*np.cos(2*np.pi*x)[:,None] - .29*np.sin(2*np.pi*x)[:,None]
        level = level + .23*np.cos(2*np.pi*y)[None,:]
        np.testing.assert_array_equal(geometry.mask(c),(level>=0).astype(np.uint8))
        np.testing.assert_array_equal(geometry.mask(c),geometry.mask(c)[:,::-1])
        assert geometry.mask(np.zeros(geometry.dimensions)).all()


def test_centered_decoder_matches_frozen_h12_source_masks():
    fixture = json.loads((Path(__file__).parent/"fixtures/h12_decoder_traces.json").read_text())
    space = MaskSpace(tuple(fixture["shape"]),decoder="h12_cell_centers_v1")
    geometries = {}
    for row in fixture["cases"]:
        mode = tuple(row["mode"])
        geom = geometries.setdefault(mode,FourierSpace(space,mode))
        c = np.random.default_rng(row["seed"]).normal(size=geom.dimensions)
        assert hashlib.sha256(geom.mask(c).tobytes()).hexdigest() == row["sha256"]


def test_covariance_refactor_caps_actual_reconstruction():
    k = 9
    u = np.zeros(k);u[:2] = 1/np.sqrt(2)
    v = np.zeros(k);v[0],v[1] = 1/np.sqrt(2),-1/np.sqrt(2)
    matrix = .05*np.eye(k)+3.95*np.outer(u,u)+2.95*np.outer(v,v)
    diagonal,low,info = covariance_refactor(matrix,1,True)
    actual = np.diag(diagonal)+low@low.T
    spectrum = np.linalg.eigvalsh(actual)
    assert info["unshrunk_max"] == pytest.approx(5.475)
    assert spectrum.min() >= .05-1e-10 and spectrum.max() <= 4+1e-10
    assert covariance_refactor(np.eye(k),0)[1].shape == (k,0)


def test_tangent_optimizer_matches_submitted_h12_synthetic_trace():
    fixture = json.loads((Path(__file__).parent/"fixtures/h12_optimizer_trace.json").read_text())
    opt = create_optimizer("tangent_covariance",MaskSpace(tuple(fixture["shape"]),
                            decoder="h12_cell_centers_v1"),{"modes_x":2,"modes_y":1},fixture["seed"])
    for row in fixture["rows"]:
        proposal = opt.propose()[0]
        mask = np.asarray(proposal.candidate,dtype=np.uint8).reshape(fixture["shape"])
        assert hashlib.sha256(mask.tobytes()).hexdigest() == row["mask_sha256"]
        value = float(.2+.5*np.mean(mask[:,:32]))
        assert value == row["score"]
        opt.observe([Observation(proposal.id,proposal.candidate,value)])
        if row["mean"] is not None:
            np.testing.assert_allclose(opt.mean,row["mean"],atol=1e-12)
            assert opt.sigma == pytest.approx(row["sigma"],abs=1e-12)
            np.testing.assert_allclose(opt.inspect()["spectrum"],row["covariance_spectrum"],atol=1e-12)


def test_tangent_covariance_invariants_after_update_and_ties():
    opt = create_optimizer("tangent_covariance",space_for("tangent_covariance"),
                           {"modes_x":2,"modes_y":1,"population":8},seed=3)
    run_steps(opt,8)
    view = opt.inspect()
    assert view["generation"] == 1 and view["informative_updates"] == 1
    assert np.linalg.norm(view["mean"]) == pytest.approx(1)
    assert min(view["spectrum"]) >= .05-1e-10 and max(view["spectrum"]) <= 4+1e-10
    tied = create_optimizer("tangent_covariance",space_for("tangent_covariance"),
                            {"modes_x":2,"modes_y":1,"population":8},seed=3)
    Runner(tied,lambda mask:.5)
    for _ in range(8):
        Runner(tied,lambda mask:.5).step()
    assert tied.inspect()["no_information_ties"] == 1
    assert tied.inspect()["informative_updates"] == 0


def test_nested_continuation_preserves_incumbent_at_lift():
    opt = create_optimizer("nested_fourier",MaskSpace((16,12)),settings("nested_fourier"),seed=4)
    runner = Runner(opt,lambda mask:0.)
    runner.step()
    incumbent = opt.geometry.mask(opt.coefficients)
    runner.step();runner.step()
    assert opt.stage == 1
    np.testing.assert_array_equal(opt.geometry.mask(opt.coefficients),incumbent)


def test_motif_edits_and_phenotype_descriptors_are_physical():
    mask = np.zeros((8,8),dtype=np.uint8)
    changed = mirrored_edit(mask,{(2,1),(3,2)},1)
    np.testing.assert_array_equal(changed,changed[:,::-1])
    assert len(components(changed,1,True)) >= 1
    assert descriptor(changed,10) == descriptor(changed.copy(),10)
    opt = create_optimizer("phenotype_de",MaskSpace((16,12)),{"modes_x":2,"modes_y":1,"population_size":4},seed=9)
    run_steps(opt,20)
    assert len(opt.archive) == opt.evaluations
    assert all(row["bin"] == bin_id for bin_id,row in opt.phenotypes.items())


def test_contract_rejects_invalid_observations_and_missing_gradients():
    opt = create_optimizer("flrl_autograd_adam",MaskSpace((16,12)),settings("flrl_autograd_adam"),seed=2)
    with pytest.raises(ValueError,match="callback"):
        Runner(opt,score).step()
    proposal = opt.pending
    with pytest.raises(ValueError,match="finite"):
        Observation(proposal.id,proposal.candidate,float("nan"))
    with pytest.raises(ValueError,match="does not match"):
        opt.observe([Observation("wrong",proposal.candidate,.5)])


def test_adam_uses_injected_relaxed_gradient():
    opt = create_optimizer("flrl_autograd_adam",MaskSpace((16,12)),
                           {"level_set_modes_x":2,"level_set_modes_y":1},seed=11)
    before = opt.c.copy()
    Runner(opt,lambda mask:.5,lambda c,beta,material_map:RelaxedEvaluation(-float(c@c),-2*c)).step()
    assert opt.gradient_calls == 1 and opt.adam_step == 1
    assert np.linalg.norm(opt.c) < np.linalg.norm(before)


def test_polish_two_local_steps_and_rollback():
    pytest.importorskip("torch")
    opt = create_optimizer("flrl_ppo_polish",MaskSpace((16,12)),settings("flrl_ppo_polish"),seed=42)
    runner = Runner(opt,lambda mask:.3,lambda c,b,m:RelaxedEvaluation(.3,np.zeros_like(c)))
    runner.step()
    before = opt.c.copy()
    for _ in range(3): runner.step()
    np.testing.assert_array_equal(opt.c,before)
    assert opt.rollback_count == 1 and opt.adam_step == 2
    assert opt.last_reward == 0
    assert opt.propose()[0].metadata["flrl_gradient"]["beta"] == 5


def test_residual_cached_return_has_signed_reward():
    pytest.importorskip("torch")
    opt = create_optimizer("flrl_ppo_residual",MaskSpace((16,12)),
                           {**settings("flrl_ppo_residual"),"gradient_radius":0,"episode_length":20},seed=42)
    runner = Runner(opt,score)
    runner.step()
    a = opt.c.copy()
    b = opt.geometry.normalize(np.random.default_rng(99).normal(size=opt.d))
    opt.next_c = b
    runner.step()
    ab = opt.last_reward
    opt.next_c = a
    runner.step()
    assert opt.last_reward == pytest.approx(-ab)
    opt.next_c = a.copy()
    runner.step()
    assert opt.last_reward == 0


def test_motif_rejects_equal_score_and_keeps_incumbent():
    opt = create_optimizer("motif_surgery",MaskSpace((16,12)),{"initial_count":2},seed=8)
    runner = Runner(opt,lambda mask:.5)
    runner.step();runner.step()
    before = opt.incumbent.copy()
    for _ in range(8): runner.step()
    np.testing.assert_array_equal(opt.incumbent,before)
    assert opt.accepted == 0


def test_phenotype_bin_retains_best_scored_representative():
    opt = create_optimizer("phenotype_de",MaskSpace((16,12)),
                           {"modes_x":2,"modes_y":1,"population_size":4},seed=12)
    observations = run_steps(opt,24)
    expected = {}
    for candidate,value in observations:
        bin_id = descriptor(np.array(candidate,dtype=np.uint8).reshape(16,12),opt.bins)
        expected[bin_id] = max(value,expected.get(bin_id,-float("inf")))
    assert set(expected) == set(opt.phenotypes)
    for bin_id,best in expected.items():
        assert opt.phenotypes[bin_id]["score"] == best
