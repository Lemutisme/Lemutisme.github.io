---
title: "Global Optimality Certification for Spatiotemporal Physics and PDEs"
description: "Exact equality elimination compresses discretized PDE optimization from thousands of variables to single digits, turning global certification into a neural verification problem - and moving the bottleneck to the output head."
publishedAt: 2026-08-30
tags:
  - Formal Verification
  - Global Optimization
  - PDEs
  - Neural Network Verification
draft: false
---

*Audit baseline: 28 August 2026 (`Grid_Transient_Reachability`, `Mod2ONNX`). All figures below trace to that audit; solver wall times and cross-solver comparisons are in the final section.*

## 1. The core paradox

In scientific computing and safety-critical engineering - transient power grid stability, fluid shock mitigation, biological excitation control - simulation-driven optimization is standard practice: sample initial conditions or control trajectories, run numerical integrators, keep the best design. This is efficient, and it produces only **candidate solutions (primal incumbents)**. It cannot produce **certificates of global optimality or safety (dual bounds)**.

Deterministic global certification requires enclosing the optimal value $f^\star$ in a rigorous interval:

$$
f^\star \in [\text{dual bound } d, \; \text{primal incumbent } p], \qquad \text{Gap} = \frac{p - d}{\vert p \vert + \epsilon}
$$

When $p - d \le \epsilon_{\text{tol}}$, global optimality is proved.

The conventional obstruction to applying spatial branch-and-bound (BaB) to discretized PDEs and differential-algebraic equations (DAEs) is dimensionality. Discretizing space and time lifts the decision space into thousands of algebraic variables ($N_x \times N_t \sim 10^3\text{–}10^4$). Splitting a 7,245-dimensional box once along *every* coordinate would yield $2^{7245}$ orthants; even though BaB splits one coordinate at a time, the search cannot make meaningful progress in a space of that width.

```
   Lifted formulation (classical BaB)            Reduced computation graph (LiRPA BaB)
   ==================================            =====================================
   • Free variables: (z, y) in R^7245            • Free decisions: z in R^2
   • Equalities: r(z, y) = 0 (implicit)          • Feedforward rollout: y = Phi(z) (explicit)
   • BaB splits: over full spacetime grid        • BaB splits: over low-dimensional input box
   • Status: no meaningful step possible         • Status: 2D plane bisection + bound propagation
```

The pivotal realization is that **state variables across spacetime are consequences, not degrees of freedom**. Eliminating physical equality constraints via symbolic DAG compilation and Schur-complement reduction compresses the search space from thousands of dimensions to single digits ($d \le 20$; $d = 2$ for the power grid DAEs).

This compression obeys a conservation law:

> Exact equality elimination trades an intractably wide input search space for an ultra-deep, wide output computation graph whose convex relaxation loosens with rollout depth.

Both halves have to be reported together. The rest of this note is largely about the second half.

## 2. Exact equality elimination

Consider a lifted spatiotemporal problem:

$$
\begin{aligned} \min_{z \in \mathcal{Z}, \, y \in \mathcal{Y}} \quad & f(z, y) \\ \text{s.t.} \quad & r_k(z, y) = 0, \quad \forall k \in \{1, \dots, K\} \\ & g_j(z, y) \le 0, \quad \forall j \in \{1, \dots, J\} \\ & \ell_z \le z \le u_z, \quad \ell_y \le y \le u_y \end{aligned}
$$

where $z \in \mathbb{R}^d$ are free physical parameters or controls and $y \in \mathbb{R}^N$ are spatial states across time levels.

### 2.1 Symbolic graph compilation

Rather than handing $r(z, y) = 0$ to a MINLP solver as algebraic constraints, the compiler runs four deterministic transformations:

```
  .mod / lifted NLP
        |
        v
  1. Symbolic linear isolation:  a * y_j + q(z, y_<j) = 0
     ==> y_j := -q(z, y_<j) / a
        |
        v
  2. Dependency DAG + Kahn topological sort
     ==> y^(1) = Phi_1(z), ..., y^(L) = Phi_L(z, y^(<L))
        |
        v
  3. Sound range clamping (HardTanh)
     ==> injects physical bounds [l_y, u_y] to arrest relaxation
         divergence during linear bound propagation
        |
        v
  4. Output head assembly
     ==> Y(z) = [objective Y_0, box rows, path rows, residuals]
        |
        v
  Reduced ONNX graph for alpha,beta-CROWN
```

The `HardTanh` nodes are per-level physical-range clipping inserted by the compiler. They are not PDE operators, and they are a different mechanism from the *complete input clipping* discussed in §5.

### 2.2 Standard verification form

The compiled problem reaches the verifier as a single computation graph $\Phi: \mathbb{R}^d \to \mathbb{R}^{m+1}$:

$$
\min_{z \in [\ell_z, u_z]} Y_0(z) \quad \text{s.t.} \quad Y_i(z) \le 0, \quad \forall i \in \{1, \dots, m\}
$$

For every definitionally eliminated instance, the output head dimension is not an independent measurement - it is fixed by the other counts:

$$
m = \underbrace{1}_{\text{objective}} + \underbrace{2 \times \vert y_{\text{elim}} \vert}_{\text{lower and upper box row per eliminated variable}} + \underbrace{N_{\text{path}}}_{\text{safety constraints}} + \underbrace{2 \times N_{\text{residual}}}_{\text{signed residual equalities}}
$$

**The factor of two dominates.** Each eliminated variable contributes both a lower and an upper bound row, so the output head grows at twice the rate the input dimension shrinks. Elimination trades a cost exponential in input dimension for one linear in output rows - an enormous net win, but a trade.

Case9 and Case57 do not follow this formula: their rollout is constructed analytically rather than by definitional elimination.

## 3. Structural census

Seven dynamical systems. Six are first-principles physics; one is an empirical neural surrogate.

| Instance | System | Lifted $(z,y)$ | Free dim $d$ | Output head $m$ | Comp. nodes | Structurally nonlinear | Status |
|---|---|---:|---:|---:|---:|---|---|
| **FHN** | FitzHugh–Nagumo PDE | 1,120 | **20** | 3,313 | 465 | 58 (12.47%) | **Certified**, $\Delta = 1.08\times10^{-6}$ |
| **Case9** | 3-machine power DAE | 1,841 | **2** | 1,243 | 1,177 | 165 (14.02%) | Feasible, open — $[0.4279,\,0.4890]$ † |
| **Case57** | 7-machine power DAE | 7,245 | **2** | **8 (aggregated)** | 1,320 | 202 (15.30%) | Nearly certified — 0.0410% **cross-solver** ‡ |
| **Burgers** | Viscous conservation PDE | 2,680 | **7** | 5,447 | 2,441 | 242 (9.91%) | Feasible, weak dual — $[-9.0,\,-0.814]$ |
| **Allen–Cahn** | Phase transition PDE | 7,640 | **7** | 15,367 | 3,878 | 481 (12.40%) | Bounding incomplete (900 s guard) |
| **Lorenz96** | Switched chaotic ODE | 2,518 | **88** | 5,420 | 2,394 | 243 (10.15%) | Numerical certificate at $10^{-4}$ only |
| **KS50** | Frozen MLP surrogate | 3,986 | **50** | 1,677 | 77 | 9 (11.69%) | Surrogate bound — **no PDE claim** |

The nominal budget was 600 s with eight Gurobi threads, but **ABCROWN wall time can exceed it** because some GPU bound-propagation calls are not interruptible. Actual times: FHN 216.81 s, Lorenz96 423.96 s, Case9 605.50 s, Case57 634.58 s, KS50 640.24 s, Burgers 874.25 s, Allen–Cahn stopped at a 900 s outer guard.

† The Case9 interval **combines endpoints from two different runs on the same graph**: the strictly replayed incumbent from the main run, and the dual bound from a naive-branching retry. The main run produced a zero root dual because an incompatible strong-branching heuristic expected unavailable `lA` tensors; the retry removed that configuration failure.

‡ The 0.0410% is a **cross-solver** gap - ABCROWN's dual $0.8657670429$ against Gurobi's validated incumbent $0.8661221557$. **Neither individual 600 s run closed its own gap**: ABCROWN's own interval was $[0.8657670429,\,0.8670120549]$, a gap of $1.245\times10^{-3}$ (0.144%).

## 4. Instance deep-dives

### 4.1 FitzHugh–Nagumo: the one fully closed certificate

FHN models electrical action potential propagation in excitable nerve or cardiac tissue over $N_x = 25$ spatial nodes and $N_t = 10$ steps ($\Delta t = 0.1$, $\Delta x = 1/24$):

$$
\begin{aligned} v^{k+1} &= v^k + \Delta t \left( D_v L_N v^k + v^k - \frac{(v^k)^3}{3} - w^k + I_k(z) \right) \\ w^{k+1} &= w^k + \Delta t \left( D_w L_N w^k + \epsilon (v^k + a - b w^k) \right) \end{aligned}
$$

with $L_N$ the discrete Laplacian under homogeneous Neumann boundaries.

- **Decisions** ($z \in \mathbb{R}^{20}$): 6 Fourier coefficients for $v^0$, 6 for $w^0$, 8 localized stimulus amplitudes $I_k$.
- **Objective:**

$$
Y_0(z) = \underbrace{-\sum_{i \in \mathcal{T}} \gamma_i (v_i^{10} - v_{\text{rest}})^2}_{\text{target tissue activation}} + \underbrace{0.006 \Vert s \Vert_2^2}_{\text{stimulus energy}} + \underbrace{0.0015 \Vert c_{\text{init}} - c_{\text{ref}} \Vert_2^2}_{\text{initial-state regularization}}
$$

- **Output head** ($m = 3{,}313$): $1 + 2{,}200$ state box rows $+ 1{,}112$ path rows.
- **Why it closes:** temporal depth is modest ($L = 10$) and strong linear diffusion contracts the reachable sets, so relaxation looseness does not compound. Certified at $\Delta = 1.08\times10^{-6}$ in 216.81 s.

Worth stating plainly: **Gurobi also certifies FHN**, at gap $1.054\times10^{-6}$ in 48.51 s. FHN demonstrates that the pipeline is sound, not that it beats a commercial global solver on this instance.

### 4.2 Case9 and Case57: analytical Schur reduction and exact aggregation

A power grid two seconds after a disturbance. Generator rotors swing against one another; we choose two dispatch levels and must hold frequency, rotor angles, and bus voltages inside limits for the whole swing while spending as little reserve as possible.

Both use 40 semi-implicit steps over two seconds ($h = 0.05$); Case9 has three dynamic resources, Case57 has seven. Two normalized dispatch variables determine

$$
p(u) = p^0 + Ru, \qquad u \in [\underline{u}, \overline{u}] \subset \mathbb{R}^2.
$$

Two exact algebraic steps make this tractable. First, the passive network is folded away analytically by a Schur complement, so electrical power becomes

$$
P_e(\delta) = \Re\!\left[ z(\delta) \odot \overline{Y_{\rm red} z(\delta)} \right].
$$

The semi-implicit swing rollout is

$$
\begin{aligned} \omega^{k+1} &= \omega^k + h M^{-1}\!\left( p(u) - P_e(\delta^k) - D\omega^k \right), \\ \widetilde{\delta}^{k+1} &= \delta^k + h \omega_b \omega^{k+1}, \\ \delta^{k+1} &= \widetilde{\delta}^{k+1} - \mathbf{1}\rho^T \widetilde{\delta}^{k+1}, \end{aligned}
$$

where the last line projects out the center-of-inertia angle.

- **Decisions** ($u \in \mathbb{R}^2$): two normalized dispatch levels.
- **Objective:** normalized reserve activation - how much reserve the chosen dispatch has to call on, scaled to the available range. Unlike the other six instances, the audit states this in words and does not give the closed form, so none is reproduced here.
- **Constraints:** generator power limits, frequency limits, pairwise angle separation, terminal frequency limits, and bus-voltage limits.
- **Output head:** $m = 1{,}243$ for Case9 (objective plus 1,242 scalar inequalities); $m = 8$ for Case57 after aggregation, below.

The reduced Gurobi reference encodings contain 1,841 variables / 2,823 rows (Case9) and 7,245 variables / 13,639 rows (Case57). The ABCROWN graph substitutes the complete rollout and leaves only $u \in \mathbb{R}^2$ free.

**Second exact step (Case57):** 6,998 scalar inequalities are grouped into seven family heads,

$$
G_r(u) = \sum_{j \in \mathcal{F}_r} \operatorname{ReLU}(g_j(u)) \le 0.
$$

Because $\operatorname{ReLU}(\cdot) \ge 0$, $G_r \le 0$ holds exactly when every member $g_j \le 0$ does. **The aggregation is exact, not a conservative relaxation.** It is why Case57 has only eight graph outputs and 17 `ReLU` nodes despite thousands of underlying constraints - and why it is the only instance whose output head is small enough for BaB to iterate freely.

That is the whole reason Case57 is closest to a certificate. It is also worth noting what the comparison shows on Case9: Gurobi's bound after 600 s was $0.0644$, against ABCROWN's $0.4279$. On these DAEs the reduced graph gives a far stronger dual than the commercial solver on the matched encoding.

### 4.3 Viscous Burgers: shock formation and boundary residuals

Steepening velocity profiles evolving into a shock on a $33 \times 81$ grid ($\Delta x = 0.0625$, $\Delta t = 0.0125$), Rusanov flux:

$$
\frac{u_i^{k+1} - u_i^k}{\Delta t} + \frac{F_{i+1/2}^k - F_{i-1/2}^k}{\Delta x} - \nu \frac{u_{i+1}^k - 2u_i^k + u_{i-1}^k}{\Delta x^2} - \sum_{q=1}^3 c_q f_q(x_i, t_k) = 0
$$

$$
F_{i+1/2}^k = \frac{(u_i^k)^2 + (u_{i+1}^k)^2}{4} - \frac{\alpha}{2}(u_{i+1}^k - u_i^k), \qquad \alpha = 1.2
$$

- **Decisions** ($z \in \mathbb{R}^7$): 3 initial Fourier amplitudes, viscosity $\nu \in [0.0025, 0.02]$, 3 forcing amplitudes.
- **Objective:** maximize localized kinetic energy at 10 checkpoints at $T = 1.0125$, i.e. $Y_0(z) = -\sum_{p=1}^{10} u(x_p, T)^2$.
- **Output head** ($m = 5{,}447$): $1 + 5{,}346$ box rows $+ 96$ path rows ($\vert u \vert \le 1.25$) $+ 4$ signed residual rows.
- **Redundant boundary residuals:** the lifted system has 2,675 equalities for 2,673 internal states. The 2 leftover boundary equations are retained as 4 signed rows. They are algebraically implied and hold to machine precision (replay violation $1.91\times10^{-16}$).
- **Where it fails:** the primal is machine-precise at $-0.813568$, the dual is $-9.0$. Eighty sequential nonlinear convection steps compound relaxation looseness at depth.

### 4.4 Allen–Cahn: the ultra-deep composition bottleneck

Phase separation in binary alloys under a cubic double-well potential, $33 \times 81$ grid, $L = 240$ composed layers:

$$
\frac{u_i^{k+1} - u_i^k}{\Delta t} - d\,\frac{u_{i+1}^k - 2u_i^k + u_{i-1}^k}{\Delta x^2} - 5\left(u_i^k - (u_i^k)^3\right) - \sum_{q=1}^3 c_q \psi_q(x_i, t_k) = 0
$$

- **Decisions** ($z \in \mathbb{R}^7$): spectral initial conditions, mobility $d$, 3 distributed forces.
- **Objective:** $Y_0(z) = -\frac{1}{10}\sum_{p=1}^{10} u(x_p, T)$.
- **Output head:** $m = 1 + 2\times 7{,}633 + 96 + 4 = 15{,}367$.
- **Where it fails:** this is the current scaling limit. Seven inputs, yet a single global bounding pass over 15,367 rows outlived the 900 s outer guard, locking the dual at the loose root estimate $-9.7309$ against a primal of $-1.0069$.

### 4.5 Switched Lorenz96: integrality via continuous relaxation

A 30-variable cyclic atmospheric model over 80 steps ($T = 0.4$), controlled by 8 continuous forces and an 80-step discrete switch $b_k \in \{0,1\}$:

$$
x_j^{k+1} = x_j^k + \Delta t \left[ (x_{j+1}^k - x_{j-2}^k)x_{j-1}^k - x_j^k + 8 + A_j^T u + b_k(-0.5 x_j^k + 2.45\phi_j) \right]
$$

- **Switching budget:** $12 \le \sum_{k=0}^{79} b_k \le 46$.
- **Dwell-time anti-chattering:** $b_k \le b_{k-1} + b_{k+1}$ and $1 - b_k \le (1-b_{k-1}) + (1-b_{k+1})$.
- **Objective:**

$$
Y_0(u,b) = \underbrace{q(x^{80})^2 + 0.012\Vert x^{80}\Vert_2^2}_{\text{terminal calming}} + \underbrace{0.04\,\frac{\Vert u \Vert_2^2}{8}}_{\text{control effort}} + \underbrace{0.08\,\frac{\sum_k b_k}{80}}_{\text{switching penalty}}
$$

- **The integrality problem:** $\alpha,\beta$-CROWN accepts only continuous inputs, so modes are relaxed to $b_k \in [0,1]^{80}$ and integrality is enforced through 80 polynomial equalities $b_k(1-b_k) = 0$, i.e. **160 signed residual rows**. Continuous bisection drives $b_k$ toward $\{0,1\}$, but on any finitely split box a product constraint retains residual violation ($\delta = 8.52\times10^{-5}$).

The result is therefore a **numerical certificate at $10^{-4}$ tolerance, not a strict certificate**. This is a formulation limit, not a tuning one: the present encoding cannot reach zero tolerance at any time budget, because a relaxed product constraint is exactly zero only in the split limit.

**On the bilinear bottleneck.** Two node counts circulate and must not be conflated. The census reports 161 *structurally nonlinear* bilinear `Mul` nodes - both operands perturbed. Live instrumentation reports 758 of 760 `BoundMul` nodes perturbed, which counts every node with *at least one* perturbed operand, including products against constants that are linear in the perturbed input. The first number sizes the genuinely bilinear relaxation problem; the second predicts propagation cost.

Replacing intermediate CROWN propagation through multiplication nodes with interval propagation preserved the numerical result, cut the full-CROWN portion from 61.29 s to 39.99 s (34.8%), and cut total wall time from 812.42 s to 423.96 s (47.8%). The remaining ~261 s child-bound round shows multiplication is not the whole problem: serial complete clipping across 5,419 constraints still dominates.

### 4.6 KS50: where the surrogate boundary sits

- **Target PDE:** forced Kuramoto–Sivashinsky, $u_t + u u_x + u_{xx} + u_{xxxx} = f(x,t)$.
- **Audited artifact:** a **frozen, pre-trained MLP** - $\mathbb{R}^{50} \to \mathbb{R}^{192} \xrightarrow{\tanh} \cdots \xrightarrow{\tanh} \mathbb{R}^{96} \to \widehat{u} \in \mathbb{R}^{25\times32}$.
- **Inputs** ($z \in \mathbb{R}^{50}$): 20 Fourier initial-condition coefficients + 30 controls.
- **Constraints** ($m = 1{,}677$): 1,600 pointwise amplitude limits $\vert \widehat{u}_i^k \vert \le 2.6$, plus spatial energy and mean bounds.

The solver establishes bounds on **the neural approximator itself** (primal $-1.2723$, dual $-2.1247$), **not on the true KS flow**. This result must not be reported as a full-physics certificate.

## 5. Bottleneck analysis

```
                        EQUALITY ELIMINATION
                                 |
        +------------------------+------------------------+
        v                                                 v
  [Gain: search space]                          [Cost: graph topology]
  R^7245  ->  R^2                               ultra-deep, wide graph
        |                                                 |
        v                                                 v
  Input BaB bisects efficiently                 compounding relaxation error
                                                + output clipping explosion
```

**1. Complete clipping is output-bound.** Standard CROWN concretizes linear bounds $a^Tz + c$ over the input box. Complete clipping instead projects onto the linearized constraint polyhedron:

$$
\min_{z \in [\ell_z, u_z]} a^Tz + c \quad \text{s.t.} \quad A_c z + b_c \le 0
$$

Each row optimizes dual multipliers $\beta$ via `_solve_dual_var`, serially. Cost therefore scales with retained output rows and nonlinear depth, far more strongly than with the number of free variables. Allen–Cahn is the clearest case: seven inputs, one bound call over 15,367 rows, budget exceeded.

**2. Relaxation loosens with depth.** Each nonlinear operator adds an enclosing convex relaxation; over $L = 80\text{–}240$ rollout levels the envelope widens layer by layer, giving order-of-magnitude dual gaps (Burgers $[-9.0, -0.81]$, Allen–Cahn root $-9.73$).

**3. Continuous integrality traps.** Enforcing $b_k(1-b_k) = 0$ inside a continuous verifier caps feasibility verification at $10^{-4}$.

### Roadmap

```
  Current limitation                            Targeted fix
  ==================                            ============
  15k+ serial constraint checks       ------>   vectorized batch clipping + GPU reduction
  output head explosion (2|y|)        ------>   exact ReLU family aggregation
  deep relaxation loosening           ------>   active-set constraint pruning during BaB
  relaxed mode residuals (1e-4)       ------>   native binary branching on b_k
```

1. **Batch complete clipping.** Refactor `_solve_dual_var` into a batched GPU kernel computing dual linesearches across constraint sets simultaneously, reusing shared backward LiRPA coefficients.
2. **Generalize exact aggregation.** Extend the Case57 seven-family pattern $\sum_{j \in \mathcal{F}_r}\operatorname{ReLU}(g_j) \le 0$ to every instance where nonnegative violation aggregation is valid.
3. **Active output-constraint set.** Maintain an active set, add violated or ambiguous rows in batches, and verify strictly satisfied interior constraints with cheap interval bound propagation instead of full CROWN.
4. **Selective CROWN.** Use CROWN only for nonlinear nodes whose intervals materially affect active constraints; IBP elsewhere.
5. **Native mixed-integer BaB.** When branching on mode variables, create child domains with $b_k$ fixed to 0 and 1 rather than bisecting a continuous interval, eliminating binding residual rows entirely.
6. **Matched SCIP harness.** See below.

## 6. Solver comparison, and what is missing

All objectives normalized to minimization; nominal 600 s, eight Gurobi threads.

| Instance | ABCROWN | Gurobi on matching graph |
|---|---|---|
| FHN | Certified: $[-1.2494657365,\,-1.2494646552]$, gap $1.08\times10^{-6}$, 216.81 s | Certified: gap $1.054\times10^{-6}$, 48.51 s |
| Case9 | dual $0.4279$ (naive retry), replayed primal $0.4890$ | primal $0.4889$, bound $0.0644$ |
| Case57 | primal $0.8670$, dual $0.8658$, gap $1.245\times10^{-3}$ | primal $0.8661$, bound $0.3075$ |
| Burgers | primal $-0.8136$, dual $-9.0$ | primal $-0.8234$, bound $-6.8447$ |
| Allen–Cahn | primal $-1.0070$, root bound $-9.7309$ | primal $-1.0068$, bound $-1.5957$ |
| Lorenz96 | primal $=$ dual $0.6797$, violation $8.52\times10^{-5}$ | no incumbent; bound $0.0120$ |
| KS50 | primal $-1.2725$, dual $-2.1247$ | no incumbent, no finite bound |

The picture is mixed and should be read that way. Gurobi certifies FHN faster. ABCROWN's dual is dramatically stronger on the two power DAEs and on Lorenz96 and KS50, where Gurobi produces no useful incumbent at all. Gurobi's Allen–Cahn bound ($-1.5957$) is far tighter than ABCROWN's stalled root bound ($-9.7309$).

**There is no matched 600 s SCIP run for these equality-eliminated graphs.** An older report mentions a SCIP Lorenz96 incumbent of $0.7426$ and a SCIP failure on a different Burgers formulation; neither is used here, because no matched logs exist. A defensible SCIP comparison requires rerunning on the exact frozen models with solver, version, threads, tolerances, and objective normalization recorded.

## 7. Takeaways

1. **Exact equality elimination is enabling, not heuristic.** It changes the geometry of spatiotemporal global optimization, turning an intractable high-dimensional implicit search into a low-dimensional explicit feedforward verification problem.
2. **The gain is a trade and must be reported as one.** Small input dimension is bought with a wider output head (two box rows per eliminated variable) and a deeper composition whose relaxation loosens with depth. The Burgers and Allen–Cahn dual gaps are the direct cost.
3. **Certificate claims need full disclosure of residuals and provenance.** Redundant boundary residuals (Burgers, $10^{-16}$) and binding integrality residuals (Lorenz96, $10^{-5}$) are different things. So is a single-run certificate versus an interval stitched from two runs (Case9) or closed against another solver's incumbent (Case57).
4. **Surrogate boundaries must stay unambiguous.** Certifying a neural surrogate (KS50) benchmarks the solver on learned weights. It is not a proof about the PDE.
5. **The next frontier is output-constraint compression.** With input dimension already in single digits, vectorizing and aggregating output-head verification is where the remaining gains are.
