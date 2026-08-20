# Reference Notes: Generative Latent Prior (GLP)

## Purpose

This document records the parts of Generative Latent Prior that are directly relevant to this project.

It exists to keep the following distinctions straight:

* the one-step denoiser of the first branch is not GLP;
* the cheap flow matcher here is not a literal GLP reproduction;
* the flow conventions used here are fixed and must not drift;
* claims must not be attributed to the paper that it did not establish;
* GLP-scale design decisions do not transfer to this compute regime unexamined.

Primary reference:

**Learning a Generative Meta-Model of LLM Activations**
Grace Luo, Jiahai Feng, Trevor Darrell, Alec Radford, Jacob Steinhardt
ICML 2026 / arXiv:2602.06964.

Official implementation:

`github.com/g-luo/generative_latent_prior`

---

# 1. Core idea

GLP learns a generative model of hidden activations produced by a language model.

Rather than representing activations only through a sparse autoencoder reconstruction, GLP attempts to learn the distribution of natural residual activations directly.

The model can then be used as an activation-space generative prior.

Applications in the paper include:

* activation generation;
* functional reconstruction;
* activation steering post-processing;
* probing / representation analysis.

---

# 2. GLP is flow matching, not ordinary DDPM denoising

The relevant GLP model uses rectified flow / flow matching.

It is not trained as:

[
D(h+\epsilon)\to h.
]

It is also not an ordinary discrete DDPM noise-prediction model.

The training objective is a time-conditioned vector-field prediction problem.

This distinction is central to our project.

---

# 3. Activation standardization

Let raw activation be:

[
h\in\mathbb R^d.
]

GLP performs elementwise standardization:

[
x_0
===

\frac{h-\mu}{\sigma}.
]

Where:

* (\mu) is per-dimension activation mean;
* (\sigma) is per-dimension activation standard deviation.

The generative model operates in standardized activation space.

This matters because residual coordinates can have substantially different scales.

---

# 4. Flow path

Sample:

[
\epsilon\sim\mathcal N(0,I)
]

and:

[
t\sim U[0,1].
]

Construct:

[
x_t
===

(1-t)x_0+t\epsilon.
]

This is a convex combination of clean standardized activation and Gaussian noise.

Convention in these notes:

[
t=0
]

means clean/data endpoint.

[
t=1
]

means Gaussian-noise endpoint.

Do not silently reverse this convention in our code.

Some scheduler libraries use the opposite parameterization.

---

# 5. Velocity target

For the straight interpolation:

[
x_t
===

(1-t)x_0+t\epsilon,
]

the path derivative is:

[
\frac{dx_t}{dt}
===============

\epsilon-x_0.
]

Therefore GLP trains a network:

[
u_\theta(x_t,t)
]

with target:

[
u^*
===

\epsilon-x_0.
]

Loss:

[
\mathcal L_{\rm FM}
===================

E
\left[
|
u_\theta(x_t,t)
---------------

(\epsilon-x_0)
|_2^2
\right].
]

The target is a **velocity vector**.

The network does not directly output reconstructed (x_0).

---

# 6. Timestep sampling

The official implementation uses uniform timestep sampling:

[
t\sim U[0,1].
]

The relevant code does not use the logit-normal timestep weighting common in some modern diffusion-training recipes.

Our cheap flow matcher intentionally follows the uniform-time formulation.

---

# 7. GLP activation modelling is tokenwise

The GLP prior models individual token activations.

The generative prior itself does not use attention across activation tokens.

It receives one activation vector and timestep conditioning.

This is important because the language model itself already contextualized the token before producing the residual activation.

Thus GLP is a generative model over the distribution of contextual hidden states, not an additional sequence model over tokens.

---

# 8. Architecture family

GLP uses stacks of Llama-style residual gated MLP blocks.

The important high-level ingredients are:

* residual MLP blocks;
* normalization;
* SwiGLU-like gating;
* explicit timestep conditioning;
* no self-attention in the activation prior.

---

# 9. GLP block structure

A conceptual block takes:

[
z\in\mathbb R^{d_{\rm model}}
]

and timestep representation:

[
e_t.
]

After normalization:

[
r=\operatorname{LN}(z).
]

Content/up branch:

[
a
=

W_{\rm up}r.
]

Gate branch:

[
g
=

W_{\rm gate}r.
]

Time projection:

[
q_t
===

W_{\rm time}e_t.
]

Timestep conditioning multiplicatively modulates the gate.

Conceptually:

[
m
=

a
\odot
\operatorname{SiLU}
(
g\odot q_t
).
]

Down projection:

[
\Delta z
========

W_{\rm down}m.
]

Residual:

[
z'
==

z+\Delta z.
]

Exact bias/normalization details should follow the implementation being referenced rather than this conceptual summary if reproducing GLP exactly.

---

# 10. Why multiplicative time conditioning matters

The same corrupted activation coordinates can require very different interpretation depending on corruption level.

For example:

[
t=0.05
]

is almost data.

Whereas:

[
t=0.9
]

is mostly Gaussian noise.

The timestep therefore modulates which MLP features should be active.

GLP injects this conditioning directly into the gated MLP computation rather than merely concatenating one scalar (t) to the activation.

---

# 11. Timestep embedding

Scalar continuous time is converted to a sinusoidal representation.

Conceptually:

[
t
\rightarrow
\phi(t)
]

where (\phi) contains sinusoidal functions at multiple frequencies.

A learned MLP then produces:

[
e_t.
]

The same timestep representation is provided to the flow blocks, with block-specific projections into the gate dimension.

Do not confuse this with language-model token positional embeddings.

It encodes **flow time / corruption level**, not sequence position.

---

# 12. GLP scaling

The published GLP models are large relative to our project.

For Llama 1B activations with residual dimension approximately 2048, the paper studies models on the order of:

* ~0.5B;
* ~0.9B;
* ~1.7B;
* ~3.3B parameters.

For Llama 8B activations, the reported GLP is approximately:

* ~3.4B parameters.

These models are not "small denoisers".

---

# 13. Width choice in GLP

A representative GLP scaling uses:

[
d_{\rm model}
\approx
2d_{\rm input},
]

with gated-MLP hidden dimension substantially wider again.

For example, for residual dimension 2048:

[
d_{\rm model}=4096,
]

[
d_{\rm mlp}=8192.
]

The authors report that width is important for generative quality.

This is relevant when interpreting failures of our much smaller model.

---

# 14. GLP training data scale

The reference GLP is trained on approximately **1 billion token activations** from FineWeb.

Documents are processed up to approximately 2048 tokens.

BOS is excluded from activation modelling.

This data regime is much larger than our corrected activation cache of approximately 4M token activations.

Therefore:

**our project is not testing whether full-scale GLP works.**

It tests whether a tiny cheap analogue retains useful behaviour.

---

# 15. GLP training regime

Reference training uses approximately:

* one pass over ~1B streamed activations;
* batch size around 4096;
* AdamW;
* learning rate around (5\times10^{-5});
* cosine schedule;
* ~1% warmup;
* bf16;
* gradient clipping in the released implementation.

The released pipeline supports producer/consumer activation caching.

These values are reference context, not mandatory hyperparameters for our cheap GPT-2 experiment.

---

# 16. Activation generation experiment

GLP can start from Gaussian noise and integrate the learned flow back to the activation distribution.

The paper compares generated activations to true activation distributions using distributional metrics such as Fréchet-style distance.

This evaluates the generative prior itself.

Our project currently does not require high-quality unconditional activation generation.

Our primary target is steering correction.

---

# 17. Number of sampling steps in GLP

The reference work evaluates different step counts for unconditional activation generation, including:

* 1;
* 4;
* 20;
* 1000.

Distributional quality improves with more integration steps and is largely near saturation by roughly 20 steps in their reported setting.

This does **not** imply that every partial-noise steering correction requires 20 evaluations.

---

# 18. GLP on-manifold steering / SDEdit idea

The steering post-processing procedure is conceptually:

1. obtain activation (h);
2. apply steering:

[
h_s=h+\alpha v;
]

3. standardize:

[
x_s
===

\frac{h_s-\mu}{\sigma};
]

4. deliberately add partial diffusion/flow noise:

[
x_{t_s}
=======

(1-t_s)x_s+t_s\epsilon;
]

5. integrate learned flow backward from (t_s) to (0);
6. denormalize;
7. inject corrected activation back into the LM.

This is analogous to SDEdit in activation space.

---

# 19. Why SDEdit-style partial noising is conceptually different from a denoiser

The old project denoiser receives:

[
h+\alpha v
]

and attempts to directly map it toward a clean activation.

GLP-style post-processing first intentionally destroys some information:

[
x_s
\rightarrow
(1-t)x_s+t\epsilon.
]

The generative prior then reconstructs a plausible activation from the partially forgotten state.

Thus:

[
t_{\rm start}
]

controls how much exact state is preserved versus regenerated.

This bottleneck does not exist in the old one-step denoiser.

---

# 20. Reference steering setting

The main GLP steering experiments use approximately:

[
t_{\rm start}=0.5
]

and:

[
20
]

flow integration steps.

Do not interpret this as proof that `.5 / 20` is universally optimal.

It is the reference configuration used in the large GLP setting.

Our project explicitly investigates a far cheaper few-evaluation regime.

---

# 21. Scheduler convention warning

The official implementation can express diffusion/flow progress through scheduler variables whose direction differs from the paper's (t) notation.

Our repository should expose the paper convention:

[
t=0:\text{data}
]

[
t=1:\text{noise}.
]

For reporting compute, use:

**NFE = actual number of flow-network evaluations.**

Do not report scheduler nominal timesteps when fewer network calls are actually executed.

---

# 22. Reverse Euler update

Under the paper convention:

[
\frac{dx}{dt}
=============

u_\theta(x,t).
]

To go from corrupted state (t_s) back to data endpoint (0):

[
t_{i+1}<t_i.
]

Explicit Euler:

[
x_{i+1}
=======

x_i
+
(t_{i+1}-t_i)
u_\theta(x_i,t_i).
]

Because:

[
t_{i+1}-t_i<0,
]

this integrates backward along the learned velocity field.

The sign is a critical implementation invariant.

---

# 23. One-step interpretation

For the straight flow path:

[
x_t=(1-t)x_0+t\epsilon,
]

ideal MSE velocity predictor:

[
u^*(x_t,t)
==========

E[\epsilon-x_0\mid x_t].
]

A single Euler jump from (t) to zero gives:

[
\hat x_0
========

## x_t

t,u^*(x_t,t).
]

Using:

[
x_t=x_0+t(\epsilon-x_0),
]

this corresponds to a conditional-mean-style estimator of (x_0).

This provides a useful interpretation for the strong one-step reconstruction performance observed in our Phase A.

However:

this does not prove that exact probability-flow integration is theoretically redundant.

---

# 24. GLP functional reconstruction

The paper evaluates reconstructed activations by replacing true activations with GLP-generated/reconstructed activations and measuring language-model loss increase.

Reported representative ΔLM-loss values for Llama 8B are approximately:

## Base model

SAE:

[
0.1976
]

GLP:

[
0.0513.
]

## Instruct model

SAE:

[
0.2224
]

GLP:

[
0.0860.
]

The reference GLP therefore preserves downstream LM function substantially better than the compared SAE reconstruction in this experiment.

---

# 25. Reference reconstruction setup

The reconstruction evaluation uses real activations, partial corruption, and reverse flow reconstruction before reinjection.

The scorer is the relevant underlying language model.

This is conceptually similar to our use of:

[
\Delta L_{\rm LM}
=================

## L_{\rm modified}

L_{\rm clean}.
]

---

# 26. SAE steering experiment in GLP

The paper evaluates SAE steering on Llama 8B Base with many randomly selected SAE directions.

The reported setup includes approximately:

* 500 random directions;
* 5 instructions per feature;
* roughly 2500 outputs;
* steering scales extending from weak to stronger-than-activation-norm regimes.

GLP post-processing improves the reported concept–fluency trade-off according to their evaluation procedure.

---

# 27. Persona steering

The paper also evaluates persona vectors such as:

* evil;
* sycophantic;
* hallucinating.

GLP post-processing can preserve fluency while allowing stronger persona-vector intervention.

This supports the idea that a generative activation prior can help outside SAE-derived directions.

---

# 28. Sentiment steering

GLP is also evaluated on DiffMean-style sentiment steering.

This experiment is especially relevant because it includes a likelihood-based quality metric rather than relying exclusively on LLM judging.

The quality metric includes conditional negative log-likelihood under the same LM.

This motivates our use of clean-model continuation NLL as a primary quality diagnostic.

---

# 29. Where GLP helps most

Reference results suggest generative correction becomes especially useful at stronger interventions, including regimes where steering contribution is comparable to or larger than the natural activation norm.

This is plausible because larger edits are more likely to leave the natural activation distribution.

Do not assume this behaviour transfers unchanged to GPT-2 or our small prior.

---

# 30. Representation/probing result

GLP internal representations are also evaluated using probing tasks.

The paper reports that scalar units in the learned generative prior can show stronger one-dimensional predictiveness than raw residual dimensions or SAE features for many binary tasks.

Dense probes remain strong for both raw and GLP representations.

A possible interpretation is that GLP reorganizes distributed semantic information into more locally decodable coordinates.

This is secondary to our steering project but relevant mechanistic context.

---

# 31. Scaling result

The paper fits a power-law-like relationship between compute/model scale and flow/diffusion loss.

Representative fit:

[
L(C)=E+A C^{-\alpha}
]

with reported parameters approximately:

[
E=0.52,
]

[
A=435.1,
]

[
\alpha=0.169.
]

The authors estimate that closing half the remaining loss gap requires a large multiplicative increase in compute.

This reinforces that generative activation modelling appears scale-sensitive.

---

# 32. Important consequence for our project

If our 16.5M flow matcher fails, the valid conclusion is:

> this cheap small flow prior did not establish the desired steering benefit under our data/compute regime.

It is **not**:

> GLP / generative activation priors do not work.

Our model is orders of magnitude smaller and trained on orders of magnitude less data.

---

# 33. Our cheap flow matcher

Our current architecture is approximately:

[
d_{\rm input}=768,
]

[
d_{\rm model}=768,
]

[
d_{\rm mlp}=1536,
]

3 residual flow blocks.

Parameter count approximately:

[
16.5\text{M}.
]

This is intentionally budget-matched to a cheap research setting.

---

# 34. Architectural difference from reference GLP

Reference GLP is much wider relative to activation dimension.

Our model does **not** use approximately:

[
d_{\rm model}=2d_{\rm input}
]

as the main reference models do.

Therefore one key cheap-GLP research question is:

> how much of the useful generative-prior behaviour survives severe width and scale reduction?

---

# 35. Data-scale difference

Reference:

[
\sim 10^9
]

activation tokens.

Our project:

[
\sim4\times10^6.
]

Ratio:

approximately hundreds-fold less training data.

This difference must be mentioned when interpreting negative results.

---

# 36. Current Phase A result

Our cheap flow model successfully reconstructs partially corrupted activations functionally.

Clean LM loss:

[
3.2723.
]

Functional reconstruction:

| t_start | corrupted ΔLM |  NFE=1 |  NFE=3 |  NFE=5 |
| ------- | ------------: | -----: | -----: | -----: |
| 0.10    |        0.0142 | 0.0068 | 0.0067 | 0.0067 |
| 0.25    |        0.1340 | 0.0537 | 0.0532 | 0.0534 |
| 0.50    |        1.3187 | 0.3610 | 0.3512 | 0.3538 |

Approximate recovered damage:

* 52%;
* 60%;
* 73%.

Thus the cheap prior is functionally meaningful.

---

# 37. Our surprising difference from reference intuition

In our Phase A:

[
NFE=1
\approx
NFE=3
\approx
NFE=5.
]

Multi-step integration buys almost nothing for functional reconstruction.

This is an empirical result of our system.

Do not overwrite it with the expectation that "diffusion needs many steps".

---

# 38. Possible explanations for NFE saturation

Current possibilities include:

1. one-step conditional-mean estimation already captures most functional reconstruction;
2. the small model is not accurate enough for repeated integration to help;
3. small-(t) velocity prediction may be weak;
4. repeated Euler calls accumulate model/discretization error;
5. the deterministic ODE sampler may not expose a benefit available to another sampler;
6. functional LM loss may saturate before distributional geometry does.

These are hypotheses.

Do not claim one has been proven.

---

# 39. What Phase B tests

Phase B asks whether SDEdit-style flow post-processing improves actual steering.

Frozen grid:

[
t_{\rm start}
\in
{0.10,0.25,0.50}
]

and:

[
NFE
\in
{1,3,5}.
]

Phase A predicts:

[
NFE
]

will matter little.

The primary variable is expected to be:

[
t_{\rm start}.
]

---

# 40. Most important Phase B control

If flow improves NLL, measure how much steering survives:

[
r_{\rm retain}
==============

\frac{
\langle\tilde h-h,v\rangle
}{
\alpha
}.
]

Then compare against ordinary additive steering or shrinkage at matched effective intervention strength.

Otherwise a method may appear to "denoise" simply because it erases the edit.

---

# 41. What would count as cheap-GLP success

Strong result:

The 16.5M flow matcher achieves lower NLL / degeneration at matched semantic concept strength than additive or scalar shrinkage.

Especially interesting if this occurs with:

[
NFE\le3.
]

This would show that some useful generative-prior behaviour survives drastic compression of GLP scale.

---

# 42. What would count as a useful negative result

If flow:

* reconstructs Gaussian/partial corruption well;
* improves nominal-alpha NLL;
* but the effect disappears after matching realised steering strength;

then the useful conclusion is:

> a small concept-agnostic flow prior can reconstruct natural activation corruption but does not automatically learn a useful nonlinear correction for structured steering perturbations.

This would motivate studying **train/test corruption mismatch**.

---

# 43. Potential structured-corruption extension

A later method may replace purely Gaussian training paths with perturbations using training-only SAE directions.

For example:

[
h+\beta w
]

with:

[
w
]

sampled from a training-only SAE direction pool.

This would test whether the failure is caused by mismatch between:

* isotropic/noise-like training corruption;
* highly structured steering perturbations.

This extension is not part of current Phase B.

---

# 44. Potential adaptive `t_start`

Another later idea is:

[
t_{\rm start}
=============

f(\hat\alpha),
]

with:

[
t_{\rm start}(0)=0.
]

Motivation:

* weak steering may require little/no correction;
* strong steering may benefit from more generative rewriting.

This would make correction strength depend on intervention strength.

Not part of current frozen Phase B.

---

# 45. Terminology rules

Use:

* **flow matcher**;
* **flow prior**;
* **cheap GLP-style prior**;
* **GLP-inspired flow matcher**;
* **SDEdit-style activation correction**.

Avoid saying:

* "our GLP" without qualification;
* "we reproduced GLP";
* "3-step diffusion model";
* "diffusion denoiser" when discussing the rectified-flow objective;
* "GLP failed" if our small model fails.

Preferred:

> 16.5M GLP-inspired time-conditioned flow matcher.

---

# 46. Reference-vs-project summary

| Property            | Reference GLP           | This project                                   |
| ------------------- | ----------------------- | ---------------------------------------------- |
| LM                  | Llama-family            | GPT-2 small                                    |
| residual dim        | 2048 / 4096             | 768                                            |
| activation data     | ~1B tokens              | ~4M tokens                                     |
| model size          | ~0.5B–3.4B              | ~16.5M                                         |
| objective           | flow matching           | flow matching                                  |
| standardization     | per-dim                 | per-dim                                        |
| prior attention     | none                    | none                                           |
| steering correction | SDEdit-style            | SDEdit-style                                   |
| reference `t_start` | ~0.5                    | sweep .1/.25/.5                                |
| reference sampling  | ~20 steps main steering | NFE 1/3/5                                      |
| current finding     | generative prior useful | Phase A reconstruction useful; Phase B pending |

---

# 47. Canonical interpretation

The project should be framed as a scale/computation question:

> Can a very small activation flow prior recover any of the steering benefits reported for large generative activation models?

The scientific value lies in separating:

* reconstruction ability;
* generative iteration;
* steering attenuation;
* nonlinear manifold correction;
* corruption-distribution mismatch;
* model/data scale.

Do not collapse these into one binary question of whether "denoising works".
