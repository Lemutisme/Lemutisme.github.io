---
title: "ShortageSim"
tagline: "Simulating how a regulatory alert moves a supply chain when nobody sees the whole picture."
description: "An LLM-agent simulation of drug shortage dynamics under information asymmetry, calibrated against a dataset of 2,925 historical FDA shortage events."
status: "AAAI 2026 Oral"
period: "2025 - 2026"
citationYear: 2026
order: 7
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

**ShortageSim: Simulating Drug Shortages under Information Asymmetry**

Mingxuan Cui\*, Yilan Jiang\*, **Duo Zhou\***, Cheng Qian, Yuji Zhang, Qiong Wang

Accepted to **AAAI 2026 (Oral)**. [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/41172) · [arXiv](https://arxiv.org/abs/2509.01813) · [Code and dataset](https://github.com/Lemutisme/ShortageSim)

## A regulatory alert works through how imperfectly informed firms read it

Drug shortages carry direct risk to patient care, and regulators respond to them largely through information: early warnings, public disclosures, and shortage alerts. Manufacturers and institutional buyers then act on those signals.

Analytical supply chain models usually assume complete information or common priors so that an equilibrium can be solved in closed form. Those assumptions remove the mechanism through which a regulatory alert actually operates: no participant sees the full state of the supply chain, and different firms draw different operational conclusions from the same public announcement.

## Three information sets and one shared announcement

ShortageSim models drug manufacturers, institutional buyers, and the regulator as interacting LLM agents whose information sets are strictly partitioned:

| Role | Observes | Does not observe |
| --- | --- | --- |
| Manufacturer | Own production capacity, cost structure, and the public FDA announcement | Competitors' capacities, which rival is disrupted, or rivals' expansion plans |
| Institutional buyer | The public announcement and historical delivery failure rates | Individual manufacturers' inventory or production schedules |
| Regulatory agency (FDA) | Aggregate shortage severity and historical shortage records | Private capital investment plans or buyer stockpiles |

The announcement is the only object all three roles can read, and each folds it into a different decision. Because the agents interpret unstructured regulatory disclosures rather than solving a common-knowledge equilibrium, the simulation can vary what a regulator discloses, how early, and with what specificity, and trace the resulting production and procurement dynamics.

<figure>
  <img src="/figures/shortagesim/architecture.png" alt="Diagram of ShortageSim. An FDA agent issues an announcement. Manufacturer and buyer agents react. Each panel lists what that role can observe and what it cannot. A timeline runs from disruption to market reaction.">
  <figcaption>Three roles, three information sets, one announcement. The simulation steps through disruption, the FDA's reading, the announcement, manufacturer and buyer decisions, and the market's response.</figcaption>
</figure>

## Calibrating against historical FDA shortage trajectories

The framework is built alongside an open dataset of 2,925 historical FDA-reported shortage events and evaluated against 51 resolved shortage trajectories. Accuracy is measured by resolution lag percentage, the relative gap between simulated and historical market clearance times. The primary evaluation focuses on discontinuation-induced shortages—where a supplier exits permanently, so market clearance depends entirely on remaining manufacturers interpreting the FDA alert and expanding capacity rather than the original plant resuming production. On that split, ShortageSim lands within a mean +4.5% of real resolution times, compared with −28.3% for a zero-shot baseline that clears shortages far too optimistically.

That shift represents an 84% reduction in simulation error: it measures how closely the multi-agent system reproduces historical clearance trajectories, not a claim that real shortages shrink by 84%.

## Why a third manufacturer changes how an alert works

Because firms act under private information, the same regulatory announcement induces different capacity investments depending on market concentration. In the simulation, average supply per manufacturer is highest in a duopoly (two firms), drops sharply once a third manufacturer is present, and stays near that lower plateau out to five or ten firms.

The information partition explains why the threshold sits between two and three firms. In a two-firm market, an intact manufacturer reading an FDA shortage alert can deduce with certainty that its sole competitor is the disrupted supplier, making capacity expansion a high-certainty return. Once a third firm enters, an intact manufacturer cannot tell which rival was disrupted or how many other intact rivals are expanding at the same time; coordination uncertainty and the risk of overcapacity make each firm invest more conservatively. A disclosure policy tuned on a duopoly therefore behaves as a different intervention in a market with three or more suppliers.

<figure>
  <img src="/figures/shortagesim/competition.png" alt="Bar chart of average supply per period against the number of manufacturers. Two manufacturers are near 1.33. Three, three-to-five, and five-to-ten manufacturers sit near 1.0, with wide intervals.">
  <figcaption>Simulated supply per manufacturer as the market gets less concentrated. The drop is between two firms and three. Adding more firms after that does not restore the duopoly level.</figcaption>
</figure>

## Where historical calibration stops

A simulation is only as reliable as the decision rules its agents stand in for. Calibrating agent behavior against recorded FDA shortage events establishes that the model can reproduce historical trajectories; it does not by itself prove that LLM agents will match how real firms under binding capital constraints react to a novel disclosure policy that has never been tried. Establishing where simulated counterfactuals remain valid and where language-model heuristics diverge from institutional incentives is the boundary between a descriptive simulator and a prescriptive policy tool.
