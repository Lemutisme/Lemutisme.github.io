---
title: "Distributionally Robust Decision Making"
tagline: "Control and reinforcement learning that hold up when the model was fit from limited data."
description: "Applying distributionally robust optimization to stochastic control and offline reinforcement learning, in the cases where the robust problem stays tractable enough to solve."
status: "ICLR 2026"
period: "2023 - 2026"
citationYear: 2026
order: 4
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

## Research question

A controller or a policy is almost always synthesized against a model that was fit to a finite dataset. Which guarantees survive when the environment it is deployed into differs from the one the data described?

Classical stochastic control and modern offline reinforcement learning share this failure mode. The nominal model is an estimate, the estimate carries sampling error, and a policy optimized aggressively against that estimate can degrade sharply under small perturbations of the true dynamics.

## Core idea

Distributionally robust optimization replaces the point estimate with an ambiguity set - a neighborhood of distributions around the empirical model - and optimizes against the worst case inside that set rather than against the estimate itself.

The difficulty is that the robust problem is usually much harder than the nominal one. This line of work looks for the cases where robustness stays affordable: where the robust formulation reduces to a structure that already has an efficient solution scheme, or where the inner worst-case problem admits a form that can be solved directly.

## Distributionally Robust Path Integral Control

The continuous-time, continuous-space setting, where the controller does not know the underlying diffusion process and has only a finite set of historical disturbance trajectories. When data collection is limited, a controller synthesized from the empirical distribution alone performs poorly.

DRPI applies distributionally robust optimization to robustify the policy against the unknown diffusion. The resulting scheme turns out to resemble risk-sensitive control, and that resemblance is what makes it practical: the path integral control framework then applies as an efficient solution method. The performance guarantees derived for DRPI align closely with the choice of a risk parameter in the risk-sensitive formulation.

## DR-SAC

Distributionally robust reinforcement learning had been largely restricted to value-based methods in tabular settings. DR-SAC is an actor-critic distributionally robust algorithm for offline learning in continuous action spaces.

The objective maximizes entropy-regularized reward against the worst transition model within a KL-divergence constrained uncertainty set. Two components make it work:

- a distributionally robust form of soft policy iteration, with a convergence guarantee; and
- a generative modeling step that estimates the unknown nominal transition model from the offline dataset.

Across continuous control benchmarks under common perturbations, the resulting policies reach up to 9.8 times the average reward of the SAC baseline.

## Open directions

The two results sit at different points on the same trade-off: how much conservatism to buy, and how much structure to give up to keep the robust problem solvable. The choice of ambiguity set is the lever - KL divergence is convenient but couples the radius to the nominal model in ways that are hard to interpret. How to select a set that reflects an actual data budget, and how these guarantees behave when the policy keeps collecting data rather than learning offline once, both remain open.
