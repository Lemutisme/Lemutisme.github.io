---
title: "ShortageSim"
tagline: "Simulating how a regulatory alert moves a supply chain when nobody sees the whole picture."
description: "An LLM-agent simulation of drug shortage dynamics under information asymmetry, calibrated against a dataset of 2,925 historical FDA shortage events."
status: "AAAI 2026 Oral"
period: "2025 - 2026"
citationYear: 2026
order: 5
tags:
  - LLM Agents
  - Simulation
  - Policy Design
  - Supply Chains
links:
  - label: "Paper - AAAI 2026"
    url: "https://ojs.aaai.org/index.php/AAAI/article/view/41172"
  - label: "arXiv"
    url: "https://arxiv.org/abs/2509.01813"
  - label: "Code and dataset"
    url: "https://github.com/Lemutisme/ShortageSim"
bibtex: |
  @inproceedings{cui2026shortagesim,
    author    = {Cui, Mingxuan and Jiang, Yilan and Zhou, Duo and Qian, Cheng and Zhang, Yuji and Wang, Qiong},
    title     = {{ShortageSim}: Simulating Drug Shortages under Information Asymmetry},
    booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
    volume    = {40},
    number    = {45},
    pages     = {38321--38330},
    year      = {2026}
  }
---

## Research question

Drug shortages carry direct risk to patient care, and regulators respond to them largely through information: early warnings, disclosures, shortage alerts. Manufacturers and institutional buyers then act on those signals.

But no participant observes the full state of the supply chain, and firms do not read the same announcement the same way. That makes the effect of any given intervention hard to establish - it runs entirely through how heterogeneous, imperfectly informed firms interpret a signal.

## Why the standard models do not answer it

Game-theoretic supply chain models generally assume perfect rationality and complete information. Those assumptions are what make the models solvable. They are also what removes the mechanism of interest: the intervention works precisely because different firms draw different conclusions from the same announcement.

## Core idea

ShortageSim models drug manufacturers and institutional buyers as LLM-based agents reacting to shortage alerts issued by a regulatory agency. Because the agents interpret announcements rather than solve for an equilibrium, the simulation can represent divergent readings of the same regulatory action and the competitive dynamics that follow from them.

This makes it possible to examine interventions that have no clean analytical counterpart - varying what a regulator discloses, how early, and with what specificity - and to observe the resulting production and procurement decisions.

## Evaluation

The framework is evaluated against 51 resolved shortage trajectories drawn from 2,925 FDA-reported events. Accuracy is measured as resolution lag percentage - how far the simulated clearance time sits from the historical one. On the discontinuation-cause split, ShortageSim lands within a mean 4.5% of real resolution times, against -28.3% for the zero-shot baseline, which clears shortages far too quickly.

The reported 84% is the reduction in that simulation error. It describes how closely the model reproduces history, not a change in how long real shortages last.

The framework and a dataset of 2,925 FDA shortage events are released openly, so that policy designs can be tested against recorded shortage history rather than only against assumptions.

## Open directions

The simulation is only as good as the behavior its agents stand in for, and calibrating that behavior against historical events is not the same as validating it against interventions that have never been tried. Establishing where the simulation's predictions can be trusted - and where an LLM agent's plausible-sounding decision diverges from what a real firm under real cost pressure would do - is the part that determines whether this is a policy design tool or a descriptive one.
