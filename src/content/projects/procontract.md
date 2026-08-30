---
title: "ProContract"
tagline: "An evidence-governed kernel for long-running and self-improving agents."
description: "ProContract separates an agent's adaptive policy from a small trusted kernel that controls obligations, evidence, and canonical state transitions."
status: "Active research"
period: "2026 - Present"
citationYear: 2026
order: 1
tags:
  - Agentic AI
  - RSI
  - AI Safety
  - Systems
links:
  - label: "Prototype"
    url: "https://github.com/Lemutisme/opencode/tree/capability-rsi"
---

## Research question

How can an AI system improve its policy, tools, and internal procedures without giving the improving component unrestricted authority over the state that defines success?

Long-running agents accumulate plans, artifacts, evaluations, and execution history. Those records describe what happened, but they do not by themselves determine whether a task is still owed, whether the supporting evidence remains valid, or whether a claimed improvement may safely become canonical.

## Core idea

ProContract treats recursive self-improvement as a separation-of-authority problem. The adaptive policy may propose changes and execute work, while a smaller kernel remains responsible for:

- representing outstanding obligations outside the executor;
- binding evidence to the exact task, revision, specification, and subject it supports;
- authorizing typed transitions into canonical state;
- preserving immutable candidate handoffs and audit receipts; and
- reopening affected obligations when accepted support is invalidated.

This decoupling allows the policy to change aggressively while the invariants governing acceptance remain stable and independently inspectable.

## System model

The current design separates three roles:

1. **Issuer** - defines an obligation and the conditions for discharge.
2. **Executor** - searches, writes code, runs experiments, and proposes candidate state changes.
3. **Institution** - validates evidence and owns the transition boundary for protected state.

The institution is deliberately narrow. It does not need to reproduce the executor's reasoning. It only needs to determine whether the submitted evidence authorizes a specific transition under the current contract.

## Why this matters for RSI

A recursively improving agent should not be trusted merely because its latest policy reports better performance. The improvement process also needs a persistent account of what must remain true across policy replacement, session loss, evaluator revision, and rollback.

ProContract is therefore not another improvement policy. It is the substrate that lets different improvement policies compete and evolve while preserving evidence, obligations, and recovery semantics.

## Current directions

- Formalizing task-standing aliasing and the limits of executor-visible state.
- Designing minimal transition rules for settlement, invalidation, and dependent reopening.
- Measuring whether kernel-governed iteration improves reliability on long-horizon coding and research tasks.
- Studying which parts of an RSI loop belong in policy and which must remain in the trusted kernel.
