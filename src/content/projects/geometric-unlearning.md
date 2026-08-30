---
title: "Geometric Language Model Unlearning"
tagline: "Separate targeted forgetting from capability preservation through optimization geometry."
description: "A geometric approach to language model unlearning that controls interference between forgetting objectives and retained behavior."
status: "EMNLP 2026 Main"
period: "2025 - 2026"
citationYear: 2026
order: 3
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
---

## Research question

Language model unlearning asks a model to remove the influence of selected data while retaining behavior that should remain intact. These objectives interact through shared parameters, so optimizing forgetting alone can damage unrelated capabilities, while strong preservation penalties can prevent effective forgetting.

The key question is therefore geometric: which update directions remove targeted behavior while minimally interfering with the subspace needed for retention?

## Core idea

Geometric-disentanglement treats forgetting and preservation as structured directions in the model's optimization landscape. Rather than balancing two scalar losses with a single fixed weight, the method reasons about how their update geometry interacts.

This view makes the central tradeoff explicit:

- move strongly in directions that support forgetting;
- suppress components that conflict with retained behavior; and
- use the optimizer-induced geometry instead of assuming every parameter direction has the same scale.

## Why optimizer geometry matters

Modern optimizers rescale parameter directions using accumulated state. A Euclidean projection can therefore be misaligned with the update that the optimizer will actually apply. Modeling the optimizer-induced metric makes the constraint on retention consistent with the effective step taken during training.

The resulting procedure aims for the steepest feasible forgetting update under a geometry-aware trust region for preservation.

## Research contribution

The project connects machine unlearning to constrained optimization in a way that is both algorithmic and interpretable. It provides a vocabulary for distinguishing three questions that are often conflated:

1. whether the forget objective decreases;
2. whether retained behavior is preserved; and
3. whether the optimizer's actual update respects the intended separation.

The work was accepted to the EMNLP 2026 Main Conference. Code and the preprint are available through the project resources.

## Current directions

- Understanding when local gradient geometry predicts long-horizon forgetting behavior.
- Extending geometric separation beyond a single forget and retain pair.
- Relating empirical unlearning metrics to stronger guarantees about information removal.
- Applying the same constrained-update viewpoint to post-training safety interventions.
