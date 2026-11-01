---
title: "Distributionally Robust Decision Making"
tagline: "Control and reinforcement learning that hold up when the model was fit from limited data."
description: "Applying distributionally robust optimization to stochastic control and offline reinforcement learning, in the cases where the robust problem stays tractable enough to solve."
status: "ICLR 2026"
period: "2023 - 2026"
citationYear: 2026
order: 6
tags:
  - Distributionally Robust Optimization
  - Reinforcement Learning
  - Stochastic Control
  - Optimization
links:
  - label: "DR-SAC - ICLR 2026"
    url: "https://openreview.net/forum?id=a19MA0ksbc"
  - label: "DR-SAC - arXiv"
    url: "https://arxiv.org/abs/2506.12622"
  - label: "DR-SAC code"
    url: "https://github.com/Lemutisme/DR-SAC"
  - label: "DRPI - ACC 2024"
    url: "https://ieeexplore.ieee.org/abstract/document/10644179"
  - label: "DRPI - arXiv"
    url: "https://arxiv.org/abs/2310.01633"
  - label: "DRPI code"
    url: "https://github.com/Lemutisme/Distributionally-Robust-Path-Integral-Control"
bibtex: |
  @inproceedings{cui2026drsac,
    author    = {Cui, Mingxuan and Zhou, Duo and Han, Yuxuan and Hanasusanto, Grani A. and Wang, Qiong and Zhang, Huan and Zhou, Zhengyuan},
    title     = {{DR-SAC}: Distributionally Robust Soft Actor-Critic for Reinforcement Learning under Uncertainty},
    booktitle = {The Fourteenth International Conference on Learning Representations},
    pages     = {92720--92750},
    year      = {2026}
  }

  @inproceedings{park2024drpi,
    author    = {Park, Hyuk and Zhou, Duo and Hanasusanto, Grani A. and Tanaka, Takashi},
    title     = {Distributionally Robust Path Integral Control},
    booktitle = {American Control Conference},
    pages     = {1164--1171},
    year      = {2024}
  }
---

**DR-SAC (ICLR 2026) · Distributionally Robust Path Integral Control (ACC 2024)**

[DR-SAC Paper](https://openreview.net/forum?id=a19MA0ksbc) · [DR-SAC Code](https://github.com/Lemutisme/DR-SAC) · [DRPI Paper](https://ieeexplore.ieee.org/abstract/document/10644179) · [DRPI Code](https://github.com/Lemutisme/Distributionally-Robust-Path-Integral-Control)

## Point-estimate dynamics fail where data are scarce

A controller or reinforcement learning policy is almost always synthesized against a transition model fit to a finite dataset. When that nominal model carries sampling error or faces environment shift at deployment, a policy optimized aggressively against the point estimate exploits errors in the fit and degrades sharply under small perturbations of the true dynamics.

Distributionally robust optimization (DRO) replaces the single empirical transition model $\hat{P}$ with an ambiguity set of plausible distributions around $\hat{P}$ and optimizes against the worst case inside that neighborhood. The central obstacle is computational: an inner minimization over an infinite-dimensional family of transition kernels generally destroys the Bellman or path-integral structure that makes the nominal problem solvable.

## Make worst-case expectations tractable via KL duality

This line of work isolates the cases where distributional robustness stays computationally affordable. When the ambiguity set is a Kullback–Leibler (KL) divergence ball of radius $\varepsilon$ around the empirical distribution $\hat{P}$, strong duality collapses the infinite-dimensional inner worst-case expectation of a cost $J$ into a scalar convex optimization over a temperature multiplier $\tau > 0$:

$$\sup_{P : \, D_{\mathrm{KL}}(P \,\Vert\, \hat{P}) \le \varepsilon} \mathbb{E}_P[J] = \inf_{\tau > 0} \left\{ \tau \log \mathbb{E}_{\hat{P}}\!\left[\exp\!\left(\frac{J}{\tau}\right)\right] + \tau \varepsilon \right\}.$$

Instead of solving for an adversarial transition kernel explicitly, the solver evaluates an exponentially tilted expectation under the nominal model $\hat{P}$. Both papers below exploit this exact reduction, first in continuous-time stochastic control and then in continuous-action offline reinforcement learning.

## DRPI: KL robustness preserves path-integral linearizability

In continuous-time, continuous-space stochastic control, the controller often does not know the underlying diffusion process and observes only a finite set of historical disturbance trajectories. Synthesizing a controller directly from the empirical disturbance distribution leaves it brittle when trajectories are scarce.

[Distributionally Robust Path Integral Control (DRPI)](https://ieeexplore.ieee.org/abstract/document/10644179) applies the KL-DRO reduction across diffusion trajectories. Because the dual objective takes the form of an exponential cumulant generating function, the robust control problem becomes mathematically isomorphic to risk-sensitive stochastic control, with the dual variable $\tau$ setting the effective risk sensitivity. In stochastic control, that exponential tilting is the exact structure required by the Feynman–Kac lemma to transform the nonlinear Hamilton–Jacobi–Bellman PDE into a linear PDE solvable by Monte Carlo path integrals. DRPI therefore buys worst-case distributional guarantees without giving up sampling-based path-integral synthesis.

The physical effect appears when a robot navigates around an obstacle under a misspecified noise model. Nominal path integral control fills the free space up to the boundary, so model mismatch pushes rollouts into the obstacle. DRPI shifts typical trajectories into a wider safety corridor whose sample cloud clears the obstacle across all 100 rollouts.

<figure>
  <img src="/figures/distributionally-robust/drpi-traj.png" alt="Simulated trajectories for a double integrator and a unicycle. Nominal path integral control spreads around an obstacle, including into it. Distributionally robust path integral control stays in a channel that clears the obstacle.">
  <figcaption>One hundred rollouts. Blue is path integral control on the empirical diffusion. Green is DRPI. The obstacle is the black box. The robust policy is the one whose sample cloud clears it.</figcaption>
</figure>

## DR-SAC: robust actor-critic learning in continuous action spaces

In reinforcement learning, distributionally robust methods had been largely confined to tabular MDPs or discrete action spaces where every state-action transition can be enumerated. [DR-SAC](https://openreview.net/forum?id=a19MA0ksbc) (ICLR 2026) extends distributionally robust offline learning to continuous state and action spaces within an actor-critic architecture.

DR-SAC maximizes entropy-regularized return against the worst transition kernel inside a KL-constrained ambiguity set. Two components make the algorithm practical on continuous control benchmarks:

- a distributionally robust soft policy iteration scheme with a formal contraction and convergence guarantee, where the inner worst-case Bellman backup reduces to a tractable scalar dual problem; and
- a conditional generative transition model that estimates the nominal dynamics $\hat{P}(\cdot \mid s, a)$ from the static offline dataset so the dual expectation can be evaluated off-policy.

Across continuous control tasks under environment shifts, DR-SAC reaches up to 9.8 times the average return of standard SAC (measured at a 30% pendulum length reduction). The broader signature is the degradation slope: as pendulum length, action noise, engine power, observation noise, joint damping, or HalfCheetah back-damping move away from nominal values, SAC's return falls steeply while DR-SAC holds a much flatter plateau.

<figure>
  <img src="/figures/distributionally-robust/drsac-robust.png" alt="Six plots of average reward against a perturbation. DR-SAC stays above SAC as pendulum length, action noise, engine power, observation noise, joint damping, and HalfCheetah back damping move away from the nominal value.">
  <figcaption>Offline policies evaluated off the nominal dynamics. Each curve is mean return over 50 episodes. The robust actor is trained against the worst transition model inside a KL ball around a generative estimate of the data, so a shift in length or damping is a direction the objective already paid for.</figcaption>
</figure>

## What the ambiguity set buys and what it hides

KL divergence is what makes both DRPI and DR-SAC computationally tractable, and it also sets their main limitation: a KL ball only reweights transitions inside the support of the nominal model $\hat{P}$, so it cannot protect against unseen transitions where $\hat{P}$ assigns zero density without a smoothed generative prior. Choosing an ambiguity radius $\varepsilon$ that reflects an actual finite-sample data budget, and adapting that radius online as an agent collects fresh trajectories rather than remaining fixed offline, are the open problems next.
