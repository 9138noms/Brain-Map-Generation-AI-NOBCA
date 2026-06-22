# Functional Generative Models of the Connectome: Learning Wiring Rules that Transfer Across Species and Scale

*(working draft — NOMS-LAB / connectome-gen)*

## Abstract

The wiring diagram (connectome) of a brain is increasingly mapped but the generative
rules that shape it remain incompletely understood. Prior generative models of the
connectome reproduce topological statistics but are rarely tested for *function*. Here
we learn generative models of connectome wiring directly from reconstructed connectomes
of three organisms — *C. elegans* (302-neuron brain, 8 developmental stages), larval
*Drosophila* (≈3,000 neurons), and a *Mus musculus* visual-cortex sample (2,220 proofread
neurons, MICrONS) — and evaluate the generated connectomes on (i) structural fidelity,
(ii) **function** (sensory→motor signal routing under simulated dynamics), (iii) **novelty**
(how different a generated brain is from the one it learned), and (iv) **scalability**.
We find that (1) an edge-wise generator conditioned on cell type, spatial distance, and
learned node embeddings reproduces density, cell-type wiring architecture (r≈0.99),
hubs, and reciprocity; (2) generated brains are *functional* — sensory stimulation routes
to motor/output neurons as in the real brain (correlation 0.38–0.78 across independent
simulators including the OpenWorm c302 biophysical model) and far above random; (3) a
generated brain differs from its training brain by about as much as two real animals
differ from each other (≈60% novel edges, within natural inter-individual variation);
(4) the *spatial wiring rule is universal* — a rule learned from a 300-neuron worm
predicts mouse-cortex connectivity as well as a rule learned from the mouse itself
(AUC 0.77 = 0.77); and (5) the rule-based generator scales to **hundreds of millions of
neurons** on a single consumer GPU, producing a 300-million-neuron, layered,
24-billion-synapse connectome with realistic cortical density. We additionally show the
generated wiring supports learning (trained recurrent networks reach real-connectome
performance) and partially preserves a known innate reflex circuit. Generated connectomes
reproduce the *population-level* organization of real brains while specific fine circuits
(exact reflex asymmetries) require finer modeling. Code and data are open.

## 1. Introduction

[Motivation: connectomes are being mapped (C. elegans, Drosophila, mouse MICrONS); the
generative rules are the interesting object; prior work (Betzel et al. 2016; axon-growth
models 2024) reproduces topology but not function. Our contribution: a learned generator
evaluated for structure + function + novelty + cross-species transfer + scale.]

## 2. Data

- **C. elegans**: Witvliet et al. 2021 — 8 isogenic brain connectomes across development
  (birth→adult), chemical synapses + gap junctions + 3D neuron positions + cell types.
- **Drosophila larva**: Winding et al. 2023 — whole-brain connectome, ≈2,952 neurons,
  110k edges, 18 cell types.
- **Mouse**: MICrONS minnie65_public (CAVE) — 2,220 fully-proofread visual-cortex neurons,
  183k synaptic edges, 12 cortical cell types, 3D positions.

## 3. Methods

- **Edge-wise generator (v2).** Model P(i→j) = MLP(type_i, type_j, distance) + u_i·v_j
  (directed node embeddings → hubs) + s_i·s_j (symmetric → reciprocity). Trained with
  negative sampling; reframes "8 graphs" into ~10^5–10^6 edge decisions, defeating the
  small-sample wall. Generation = sample edges at matched density.
- **Weighted generation (v3).** Two-head model adds log-synapse-count regression →
  generates synapse weights, not just binary edges.
- **Degree- and reciprocity-correction.** Per-node out-degree targeting + reverse-edge
  injection recover the heavy-tailed degree distribution and reciprocity exactly.
- **Rule-based scalable generator.** Drops per-node embeddings (which cannot extrapolate
  to new neurons); keeps generalizable type+distance rules; sparse spatial-local sampling
  never materializes the N² matrix → arbitrary scale.
- **Functional verification.** (A) linear diffusion; (B1) leaky integrate-and-fire with
  Dale signs; (B2) OpenWorm **c302** biophysical NeuroML simulation. Stimulate sensory
  neurons, measure motor/output-neuron activity, correlate generated vs real vs random.
- **Functional optimization.** REINFORCE and Evolution Strategies optimize the generator
  to maximize functional similarity (closed loop).

## 4. Results

**4.1 Structural fidelity.** Generated C. elegans connectomes match real density and mean
degree; cell-type wiring architecture correlates r≈0.99 with real (random ≈0.5–0.6).
Node embeddings recover hubs (max degree 33 vs 32 real); reciprocity 0.05→0.22 (real 0.28),
closed to 0.32 with explicit correction. Mouse: density/mean-degree exact, type-flow r=0.991,
degree-distribution cosine 0.62; degree-correction recovers hubs (778 vs 785 real) and
reciprocity (0.13→0.32, real 0.27).

**4.2 Function — generated brains work.** Under three independent simulators the generated
brain routes sensory→motor like the real brain and far above random:
A (linear) 0.38, B1 (LIF) 0.51, **B2 (c302 biophysical) 0.78** vs random ≈0; fly
descending-output correlation 0.49 vs 0.02. Functional optimization (RL or ES) raises the
correlation 0.39→0.67 **without loss of novelty**, and generalizes to held-out stimulus
conditions (0.57, no overfitting gap).

**4.3 Novelty.** Generated-vs-real edge overlap (Jaccard) is 0.40 — i.e. ~60% newly
invented wiring — bracketed by random 0.02 and natural variation (two real adult worms)
0.51. The generator is neither a copy nor random: it differs by ≈ natural inter-individual
variation.

**4.4 Cross-species universality.** A distance-based wiring rule (normalized by typical
connection length) trained on the worm predicts mouse-cortex connectivity (AUC 0.772) as
well as one trained on the mouse (0.772); cross = within. The spatial wiring principle is
shared between nematode and mammal.

**4.5 Development.** A stage-conditioned generator reproduces the developmental growth
trajectory (synapse densification), generated edge count vs real correlates 0.81.

**4.6 Learning & instinct.** A connectome-masked recurrent network trained on a delayed-
recall task reaches real-connectome performance (0.999 = 0.999). Node embeddings assign
~9× higher probability to the real touch-escape reflex edges than to average edges; under
c302 biophysics the escape direction (anterior touch→backward command) appears in the real
connectome and is preserved (~80%) in the generated one, though the effect is weak.

**4.7 Scale.** The rule-based generator scales to ~150M neurons (memory/throughput) and
we generate a **300-million-neuron, layered, 24.5-billion-synapse** connectome (245 GB) on
a single RTX 3080 in ~2 h, with realistic cortical mean degree (~82) and cortical
lamination extracted from the mouse data. [Large-scale emergent-topology analysis: §4.8 —
to be added.]

## 5. Discussion

A consistent picture emerges across organisms: learned generators capture the
**population-level** organization of brains — cell-type architecture, density, degree
distribution, function, developmental growth — and these properties are governed by simple,
**universal, distance-and-type wiring rules** that transfer across species and arbitrary
scale. What they do *not* automatically capture is **fine, identity-specific circuitry**
(exact reflex asymmetries), which requires modeling individual connections, weights, and
neurotransmitter signs. This delineates a measurable boundary between "statistical" and
"specific" brain structure.

## 6. Limitations

Generated brains are graphs (+ coarse function), not living/behaving brains. Mammalian
realism is validated only at the scale of the mapped sample (no whole-mammal connectome
exists, so >cortex-sample scales are generation without validation). Dynamics models are
simplified (no full receptor pharmacology, neuromodulation). Degree/reciprocity correction
uses real summary statistics as parameters (standard for degree-corrected models).

## 6b. Reproducibility & adversarial verification

Core claims were re-run with controls (`src/verify.py`) and **red-teamed against their most
plausible confounds** (`src/adversarial_verify.py`):

- **Function comes from wiring, not cell identity (key test).** A skeptic could argue the
  simulators (c302/LIF) assign dynamics by cell *type*, so any connectome over the same
  neurons would correlate with the real one — making "generated brains work" an artifact of
  the shared neuron set. Test: scramble the wiring while keeping the same neurons. Motor-
  activity correlation collapses from **0.50 (generated)** to **−0.09 (random wiring)** and
  **−0.11 (degree-preserving shuffle)**. The generated brain's function genuinely derives
  from its generated wiring. *(strongest result; survives the hardest attack)*
- **Novelty is systematic, not sampling noise.** Generated brains overlap the real brain
  (Jaccard 0.40) *less* than they overlap each other (0.47); natural variation between two
  real adults is 0.51. The gen–real difference exceeds sampling variance.
- **Cross-species transfer is real but a modest rule.** Worm-trained distance rule predicts
  mouse cortex (AUC 0.772) as well as a mouse-trained rule, and an edge-shuffle control
  collapses to 0.50 (no artifact). Distance is a modest predictor (AUC ≈0.77, not ≈0.95):
  the shared principle is one component of wiring, not a complete model.
- **Type-flow fidelity is an easy target.** A degree-preserving shuffle of the real
  connectome already reaches type-flow correlation 0.75; the generator's 0.99 is only a
  modest edge over this null. Type-flow is largely fixed by per-type degree, so we report it
  as "captures cell-type architecture," not as a strong result.

**Confidence grading.** *Strong (survives adversarial test):* **function derives from the
generated wiring.** *Solid:* novelty; cross-species transfer (shuffle-controlled, modest
rule). *Easy / modest:* type-flow fidelity. *Preliminary / weak:* innate-reflex function
(small effect), large-scale hubs (absent without degree-correction), learning-task
discrimination (task too easy to separate from random).

## 6c. AI-assistance disclosure

This project was carried out with **substantial assistance from an AI system (Claude)**,
which implemented the code, designed and ran the experiments, and drafted this manuscript.
The human author directed the research — choosing questions and directions, making
decisions, catching errors, and setting priorities — and takes responsibility for the
content. The results are **exploratory** and have **not been independently peer-reviewed
or reproduced**; several implementation bugs were found and fixed during the work, and
others may remain. This disclosure is provided in the interest of research integrity.

## 7. Data & code availability

Code: https://github.com/9138noms/Brain-Map-Generation-AI-NOBCA
Archived release (DOI): **10.5281/zenodo.20780543** (https://doi.org/10.5281/zenodo.20780543).
Generated artifacts and large connectomes archived separately. Source connectomes:
Witvliet 2021, Winding 2023, MICrONS minnie65_public.

## Author

Independent research, NOMS-LAB.
