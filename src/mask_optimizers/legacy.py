"""Shared Fourier search lifecycle for random, ES, PPO, Adam and reviewed hybrids.

Every physical operation is requested from the worker's evaluator. Optimizers
never call MEENT themselves, including local polishing and cached-mask returns.
"""
from copy import deepcopy

import numpy as np

from .contracts import content_hash, Proposal
from .fourier_legacy import FourierGeometry, coefficient_matrix


def tangent(c, value):
    return value - c * np.dot(c, value)


def tangent_frame(c, direction, recent, width):
    """Deterministic Gram-Schmidt with fixed coordinate fallbacks."""
    columns = []
    for vector in [direction, recent, *np.eye(len(c))]:
        vector = tangent(c, np.array(vector, copy=True))
        for column in columns:
            vector -= column * np.dot(column, vector)
        norm = np.linalg.norm(vector)
        if norm > 1e-10:
            columns.append(vector / norm)
        if len(columns) == width:
            return np.stack(columns, axis=1)
    raise ValueError('Insufficient tangent dimensions')


class FourierOptimizer:
    supports_failure_observations = False

    def __init__(self, name, instance, parameters, seed):
        self.name, self.instance, self.config = name, instance, dict(parameters)
        self.identity = content_hash([name, instance.evaluation_identity, parameters])
        self.rng = np.random.default_rng(seed)
        p = instance.configuration
        self.space = instance.space
        if self.space.decoder != 'flrl_endpoints_v1':
            raise ValueError('The ported FLRL methods require flrl_endpoints_v1')
        self.geometry = FourierGeometry(parameters.get('level_set_modes_x', 8), parameters.get('level_set_modes_y', 4), p['grid_x'], p['grid_y'])
        self.d = self.geometry.dimensions
        self.norm = 'rms' if name == 'flrl_ppo_polish' else 'l2'
        self.c = self._initial()
        self.current_score, self.current_metrics = None, np.zeros(4)
        self.best_score, self.best_c, self.best_metrics = None, None, np.zeros(4)
        self.pending = None
        self.sequence = self.evaluations = self.decisions = self.episode_step = self.episodes = 0
        self.horizon = parameters.get('episode_length', 2 if name == 'flrl_ppo_polish' else 128)
        self.frames = parameters.get('stacked_observations', 4)
        self.history = [np.zeros(self.d) for _ in range(self.frames - 1)] + [self.c.copy()]
        self.phase = 'initial'
        self.m, self.v, self.adam_step = np.zeros(self.d), np.zeros(self.d), 0
        self.recent = np.zeros(self.d)
        self.gradient = np.zeros(self.d)
        self.adam_direction = np.zeros(self.d)
        self.relaxed = np.zeros(2)
        self.last_reward = 0.
        self.last_mask_changed = False
        self.stagnation = 0
        self.gradient_calls = self.rollback_count = 0
        self.beta = parameters.get('initial_beta', 5.)
        self.population = []
        self.es_mean = self.c.copy()
        self.es_std = np.full(self.d, parameters.get('sigma', .2))
        self.ppo = None
        if name in {'flrl_ppo', 'flrl_ppo_polish', 'flrl_ppo_residual'}:
            from .ppo import PPO
            config = {**parameters, 'episode_length': self.horizon, 'complete_rollouts': name != 'flrl_ppo'}
            actions = min(parameters.get('tangent_dimensions', 4), self.d - 1) if name == 'flrl_ppo_residual' else self.d
            self.ppo = PPO(len(self._features()), actions, seed, config, cnn=name == 'flrl_ppo',
                           frames=self.frames if name == 'flrl_ppo' else 1)

    def _initial(self):
        if self.name == 'flrl_ppo':
            return np.zeros(self.d)
        c = self.rng.normal(size=self.d)
        if self.name == 'flrl_autograd_adam':
            c = np.round(np.tanh(c), 2)
        return self.geometry.normalize(c, self.norm)

    def _features(self):
        if self.name == 'flrl_ppo':
            return np.concatenate(self.history)
        if self.name == 'flrl_ppo_polish':
            return np.r_[self.c, self.current_metrics, self.geometry.nx, self.geometry.ny, (self.horizon - self.episode_step) / self.horizon]
        return np.r_[self.c, self.current_metrics, self.best_metrics, self.relaxed,
            self.current_metrics[1:3] - self.relaxed, self.last_reward / 100, float(self.last_mask_changed),
            self.beta, self.gradient / max(np.linalg.norm(self.gradient), 1e-12), self.adam_direction,
            self.m, self.v, self.recent, self.episode_step / self.horizon,
            self.stagnation / max(1, self.config.get('restart_patience', 128)),
            min(1., self.decisions / self.config.get('relaxation_stage_steps', 128))]

    def _adam(self, gradient, learning_rate):
        self.adam_step += 1
        self.m = .9 * self.m + .1 * gradient
        self.v = .999 * self.v + .001 * gradient ** 2
        return learning_rate * (self.m / (1 - .9 ** self.adam_step)) / (np.sqrt(self.v / (1 - .999 ** self.adam_step)) + 1e-8)

    def _reset_moments(self):
        self.m.fill(0); self.v.fill(0); self.adam_step = 0

    def _prepare_residual(self, metadata):
        radius = self.config.get('gradient_radius', .1)
        self.adam_direction = np.zeros(self.d)
        self.gradient = np.zeros(self.d)
        if radius:
            reply = metadata['flrl_gradient']
            self.gradient = tangent(self.c, np.array(reply['gradient']))
            self.relaxed = np.array([reply['te'], reply['tm']])
            b = tangent(self.c, self._adam(self.gradient, 1.))
            length = np.linalg.norm(b)
            if length > 1e-14:
                self.adam_direction = radius * np.tanh(self.config.get('adam_learning_rate', .05) * length / radius) * b / length
        else:
            self.relaxed = np.zeros(2)
        width = min(self.config.get('tangent_dimensions', 4), self.d - 1)
        self.frame = tangent_frame(self.c, self.adam_direction, self.recent, width)
        self.sample = self.ppo.sample(self._features())
        self.sample['tangent_frame'] = self.frame.copy()
        residual = self.frame @ (self.config.get('residual_radius', .1) / np.sqrt(width) * np.tanh(self.sample['action']))
        self.next_c = self.geometry.normalize(self.c + self.adam_direction + residual, fallback=self.c)

    def propose(self, max_candidates=1):
        if self.pending is not None or max_candidates < 1:
            raise ValueError('Observe the outstanding Fourier proposal before requesting another')
        name, c, gradient = self.name, self.c, False
        material_map = 'index'
        if name == 'flrl_lsf_random':
            c = self.geometry.normalize(self.rng.normal(size=self.d))
        elif name == 'flrl_lsf_es':
            c = self.geometry.normalize(self.es_mean + self.rng.normal(size=self.d) * self.es_std)
        elif name == 'flrl_autograd_adam':
            epochs = self.config.get('reference_epochs', 500)
            epoch = self.decisions % epochs
            if epoch == 0 and self.decisions:
                self.c = self._initial(); self._reset_moments()
            self.beta = self.config.get('initial_beta', 5.) + min(1., epoch / epochs) * (self.config.get('final_beta', 10.) - self.config.get('initial_beta', 5.))
            # Released Adam normalizes the full reconstructed coefficient matrix
            # after the first epoch, with the normalization scale detached.
            self.gradient_scale = max(np.linalg.norm(coefficient_matrix(self.c, self.geometry.nx, self.geometry.ny)), 1e-12) if epoch else 1.
            c, gradient = self.c / self.gradient_scale, True
        elif name == 'flrl_ppo' and self.current_score is not None:
            self.sample = self.ppo.sample(self._features())
            action = np.clip(self.sample['action'], -10, 10) * self.config.get('action_scaling', .01)
            c = self.geometry.normalize(self.c + action, fallback=self.c)
        elif name == 'flrl_ppo_polish' and self.current_score is not None:
            material_map = 'permittivity'
            if self.phase == 'initial':
                self.sample = self.ppo.sample(self._features())
                action = self.sample['action'].astype(float)
                action *= min(1., self.config.get('action_radius', .1) / max(np.linalg.norm(action), 1e-12))
                self.next_c = self.geometry.normalize(self.c + action, 'rms', fallback=self.c)
                self.winner = (self.current_score, self.c.copy(), self.current_metrics.copy())
                self._reset_moments()
                self.phase = 'q'
            c = self.next_c
            gradient = self.phase in {'q', 'c1'}
            self.beta = 5. if self.phase == 'q' else 10.
        elif name == 'flrl_ppo_residual':
            if self.current_score is not None:
                c = self.next_c
            gradient = self.config.get('gradient_radius', .1) > 0
            material_map = 'permittivity'
        self.proposed_c = np.array(c, copy=True)
        self.sequence += 1
        metadata = {'flrl_method': self.name, 'phase': self.phase}
        if gradient:
            metadata['flrl_gradient'] = self.geometry.gradient_request(c, self.beta, material_map)
        mask = np.asarray(self.geometry.mask(c), dtype=np.uint8).reshape(self.space.shape[1], self.space.shape[0]).T
        self.pending = Proposal(id=f'proposal_{self.sequence}', candidate=mask.reshape(-1).tolist(), metadata=metadata)
        return [self.pending]

    def _transition(self, c, metrics):
        old_c, old_score = self.c.copy(), self.current_score
        self.last_mask_changed = self.geometry.mask(old_c) != self.geometry.mask(c)
        reward = self.config.get('reward_scaling', 100.) * (metrics[0] - old_score)
        if not self.last_mask_changed:
            reward = 0.
        self.last_reward = float(reward)
        self.c, self.current_score, self.current_metrics = np.array(c, copy=True), float(metrics[0]), metrics.copy()
        if reward > 0:
            self.recent = self.c - old_c
        if self.name == 'flrl_ppo_residual':
            self.m = tangent(self.c, self.m)
        self.history = self.history[1:] + [self.c.copy()]
        self.episode_step += 1; self.decisions += 1
        done = self.episode_step >= self.horizon
        restart = self.name == 'flrl_ppo_residual' and self.stagnation >= self.config.get('restart_patience', 128)
        self.ppo.record(self.sample, reward, self._features(), done or restart,
                        normalize_reward=self.name == 'flrl_ppo', time_limit=self.name == 'flrl_ppo')
        if done or restart:
            self.episodes += 1; self.episode_step = 0
            self.c = self.best_c.copy() if restart and self.best_c is not None else self._initial()
            self.current_score = None
            self.history = [np.zeros(self.d) for _ in range(self.frames - 1)] + [self.c.copy()]
            self._reset_moments(); self.recent.fill(0); self.stagnation = 0
            if self.name == 'flrl_ppo_residual':
                stage = self.decisions // self.config.get('relaxation_stage_steps', 128)
                self.beta = self.config.get('initial_beta', 5.) if stage == 0 else self.config.get('final_beta', 10.)
        self.phase = 'initial'

    def observe(self, observations):
        if len(observations) != 1 or self.pending is None or observations[0].proposal_id != self.pending.id:
            raise ValueError('Observation does not match the pending Fourier proposal')
        obs = observations[0]
        if list(obs.candidate) != self.pending.candidate:
            raise ValueError('Observation candidate changed')
        score = obs.score
        te, tm = obs.metrics.get('te', score), obs.metrics.get('tm', score)
        metrics = np.array([score, te, tm, min(te, tm)])
        self.evaluations += 1
        if 'flrl_gradient' in self.pending.metadata:
            self.gradient_calls += 1
        if self.best_score is None or score > self.best_score:
            self.best_score, self.best_c, self.best_metrics = score, self.proposed_c.copy(), metrics.copy()
            self.stagnation = 0
        else:
            self.stagnation += 1
        name = self.name
        if name == 'flrl_lsf_random':
            self.decisions += 1
        elif name == 'flrl_lsf_es':
            self.population.append((score, self.proposed_c.copy()))
            size = self.config.get('population_size', 16)
            if len(self.population) >= size:
                elite = sorted(self.population + [(self.best_score, self.best_c)], key=lambda item: item[0], reverse=True)[:self.config.get('elite_count', 4)]
                vectors = np.stack([row[1] for row in elite])
                self.es_mean = self.geometry.normalize(vectors.mean(axis=0), fallback=self.best_c)
                self.es_std = np.maximum(.005, .8 * self.es_std + .2 * vectors.std(axis=0))
                self.population = []
            self.decisions += 1
        elif name == 'flrl_autograd_adam':
            epoch = self.decisions % self.config.get('reference_epochs', 500)
            fraction = min(1., epoch / max(1, self.config.get('reference_epochs', 500) // 2))
            rate = self.config.get('initial_lr', .2) + fraction * (self.config.get('final_lr', .05) - self.config.get('initial_lr', .2))
            gradient = np.array(obs.metadata['flrl_gradient']['gradient']) / self.gradient_scale
            self.c += self._adam(gradient, rate)
            self.decisions += 1
        elif self.current_score is None:
            self.c, self.current_score, self.current_metrics = self.proposed_c.copy(), score, metrics.copy()
            if name == 'flrl_ppo_residual':
                self._prepare_residual(obs.metadata)
        elif name == 'flrl_ppo_polish':
            if score > self.winner[0]:
                self.winner = (score, self.proposed_c.copy(), metrics.copy())
            if self.phase in {'q', 'c1'}:
                gradient = np.array(obs.metadata['flrl_gradient']['gradient'])
                self.next_c = self.geometry.normalize(self.proposed_c + self._adam(gradient, .05), 'rms', fallback=self.proposed_c)
                self.phase = 'c1' if self.phase == 'q' else 'c2'
            else:
                if self.winner[0] <= self.current_score:
                    self.rollback_count += 1
                self._transition(self.winner[1], self.winner[2])
        else:
            self._transition(self.proposed_c, metrics)
            if name == 'flrl_ppo_residual' and self.current_score is not None:
                self._prepare_residual(obs.metadata)
        self.pending = None

    def inspect(self):
        return {'phase': self.phase, 'decisions': self.decisions, 'evaluations': self.evaluations,
            'gradient_evaluations': self.gradient_calls, 'training_updates': self.ppo.updates if self.ppo else 0,
            'updates': self.ppo.updates if self.ppo else self.adam_step,
            'last_reward': self.last_reward, 'best_binary_mean': self.best_score, 'episode_step': self.episode_step,
            'rollbacks': self.rollback_count, 'coefficient_dimensions': self.d, 'beta': self.beta,
            'profile': 'ported FLRL adaptation; performance and paper reproduction unmeasured'}

    def checkpoint(self):
        state = {key: value for key, value in self.__dict__.items() if key not in {'geometry', 'instance', 'ppo'}}
        return deepcopy({'identity': self.identity, 'state': state, 'ppo': self.ppo.state_dict() if self.ppo else None})

    def restore(self, state):
        if state['identity'] != self.identity:
            raise ValueError('Fourier optimizer checkpoint belongs to a different method, configuration or evaluator')
        self.__dict__.update(deepcopy(state['state']))
        if self.ppo:
            self.ppo.load_state_dict(state['ppo'])

    def export_artifacts(self):
        return [{'kind': 'learner_diagnostics', 'media_type': 'application/json', 'data': self.inspect()}]

    def close(self):
        pass
