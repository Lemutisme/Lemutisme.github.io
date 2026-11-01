---
title: "Geometric Language Model Unlearning"
tagline: "Separate targeted forgetting from capability preservation through optimization geometry."
description: "A geometric approach to language model unlearning that controls interference between forgetting objectives and retained behavior."
status: "EMNLP 2026 Main"
period: "2025 - 2026"
citationYear: 2026
order: 5
tags:
  - Machine Unlearning
  - Language Models
  - Optimization Geometry
  - Trustworthy AI
links:
  - label: "Paper"
    url: "https://arxiv.org/abs/2511.17100"
  - label: "Code"
    url: "https://github.com/Lemutisme/Geometric-Unlearning"
bibtex: |
  @inproceedings{zhou2026geometric,
    author    = {Zhou, Duo and Zhang, Yuji and Wei, Tianxin and Qiu, Ruizhong and Yang, Ke and Lin, Xiao and Qian, Cheng and He, Jingrui and Tong, Hanghang and Ji, Heng and Zhang, Huan},
    title     = {Geometric-Disentanglement Language Model Unlearning},
    booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
    year      = {2026}
  }
---

**Geometric-Disentanglement Language Model Unlearning**

**Duo Zhou\***, Yuji Zhang\*, Tianxin Wei, Ruizhong Qiu, Ke Yang, Xiao Lin, Cheng Qian, Jingrui He, Hanghang Tong, Heng Ji, Huan Zhang

Accepted to **EMNLP 2026 Main**. [arXiv](https://arxiv.org/abs/2511.17100) · [Code](https://github.com/Lemutisme/Geometric-Unlearning)

## Scalar loss weights cannot separate shared parameters

Language model unlearning asks a model to erase the influence of a targeted forget set while leaving retained knowledge and general utility intact. Standard methods combine a forgetting loss $\mathcal{L}_f$ and a retention loss $\mathcal{L}_r$ into a single scalar objective $\mathcal{L}_f + \lambda \mathcal{L}_r$.

When the two gradients $g_f = \nabla_\theta \mathcal{L}_f$ and $g_r = \nabla_\theta \mathcal{L}_r$ share parameter coordinates and point into opposing half-spaces ($g_f^\top g_r < 0$), tuning a scalar weight $\lambda$ only slides along a one-dimensional conflict: a small $\lambda$ damages retained capabilities, while a large $\lambda$ stalls forgetting. Resolving that interference requires operating on the direction of the update rather than on the weight of the loss.

## Euclidean orthogonality fails under adaptive optimizers

A natural fix from multi-task learning is to project the forget gradient onto the orthogonal complement of the retain gradient so that $g_\perp^\top g_r = 0$. In language model fine-tuning, that Euclidean projection fails because modern adaptive optimizers (such as AdamW) rescale coordinates by an accumulated diagonal preconditioner $D_t^{-1}$ before applying the step:

$$\Delta \theta_t = -\eta \, D_t^{-1} \tilde{g}_t.$$

Even if a projected gradient $g_\perp$ satisfies $g_\perp^\top g_r = 0$ in Euclidean space, the actual parameter step $\Delta \theta_t = -\eta D_t^{-1} g_\perp$ changes the retain loss to first order by

$$\mathcal{L}_r(\theta_t + \Delta \theta_t) - \mathcal{L}_r(\theta_t) \approx -\eta \, g_r^\top D_t^{-1} g_\perp \neq 0.$$

Coordinate-wise adaptive scaling rotates the update back into the retain subspace. Geometric-Disentanglement Unlearning (GU) instead defines the trust region and the orthogonal projection inside the metric induced by the optimizer's own preconditioner $D_t^{-1}$, ensuring that the forgetting component of the actual step $\Delta \theta_t$ is first-order non-interfering with retained behavior.

<figure>
  <img src="/figures/geometric-unlearning/geometry.png" alt="Diagram comparing ordinary gradient unlearning with geometric unlearning. The ordinary step adds forget and retain gradients that are not orthogonal. Geometric unlearning projects the forget gradient onto the complement of the retain subspace before the step.">
  <figcaption>The training loss stays the same. The step changes. A baseline adds the forget and retain gradients even when they point into each other. GU routes the forget gradient through the orthogonal complement of the retain subspace, in the metric the optimizer actually uses, and leaves the retain component to do its own work.</figcaption>
</figure>

## Moving off the one-dimensional forget-retain frontier

Because GU modifies the update geometry rather than the loss function, it attaches directly to existing unlearning objectives (such as GradDiff, NPO, and SimNPO). It separates three questions that scalarized objectives conflate:

1. whether the forget objective decreases;
2. whether retained behavior is preserved; and
3. whether the optimizer's preconditioned step respects the separation between the two.

Accepted to the **EMNLP 2026 Main Conference**, the method was evaluated across 72 configurations spanning multiple LLM backbones, deletion scales, and base objectives. On the TOFU benchmark, attaching the optimizer-metric projection to SimNPO improves forgetting efficacy by up to 62%. Across all 72 configurations, retain quality is preserved or improved in 66, and general model utility in 62. In the tradeoff plots below, each arrow compares a baseline objective before and after adding the geometric projection: arrows move left (stronger forgetting) without dropping vertically (no sacrifice in retain performance, privacy, or utility).

<figure>
  <img src="/figures/geometric-unlearning/tradeoff.png" alt="Three scatter plots on TOFU. Arrows run from a baseline circle to a GU triangle. The horizontal axis is forgetting, lower better. The vertical axes are retained knowledge, privacy, and model utility, higher better.">
  <figcaption>TOFU. Each arrow is one baseline before and after the same projection. A useful arrow moves left without falling. The projection is attached to the existing method. It does not replace the forget loss.</figcaption>
</figure>

## Scope and open questions

GU enforces local first-order orthogonality at each step using the current retain batch. Over long unlearning horizons, small higher-order curvature errors can still accumulate if the retain subspace rotates, a problem addressed from a global path perspective in [Cumulative Fisher Protection](/projects/cumulative-fisher/). Together with [Gradient Interference in Group-Aware RLHF](/projects/gradient-interference-rlhf/), these works show that post-training tradeoffs in language models are governed by the geometry of parameter updates rather than scalar loss weights.
