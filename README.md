# 뇌지도 생성 AI 실험 (NOBCA)
### Brain-Map Generation AI Experiment · NOBCA = NOms Brain Connecting AI
*Functional Generative Models of the Connectome*

Learning generative models of brain wiring (connectomes) directly from reconstructed
connectomes, and evaluating the generated brains for **structure, function, novelty,
cross-species transfer, and scale** — from a 300-neuron worm to a 300-million-neuron
synthetic cortex.

> ⚠️ **Status: exploratory, AI-assisted research (not peer-reviewed).** See the disclosure
> below and in [`PAPER.md`](PAPER.md). Core claims are verified with controls; some results
> are preliminary and clearly graded.

---

## What it does

Given real connectomes, an edge-wise generative model learns *wiring rules*
— `P(neuron i → neuron j | cell types, 3D distance, learned node embeddings)` — and
generates **novel** connectomes that can be checked against reality.

## Key results (verified — see `src/verify.py`)

| Claim | Result | Control |
|---|---|---|
| **Structural fidelity** | cell-type wiring r = **0.990** ± 0.002 | random 0.63 |
| **Function** (sensory→motor routing) | **0.78** (OpenWorm c302 biophysical) | random ≈ 0 |
| **Novelty** | 60% new wiring (overlap 0.40) | natural variation 0.51 |
| **Cross-species universality** | worm-rule → mouse AUC **0.772 = 0.772** mouse-rule | edge-shuffle 0.50 |
| **Scale** | generated **300M-neuron, 24.5B-synapse** layered cortex (single RTX 3080) | — |

The standout: a spatial wiring rule learned from a **300-neuron worm predicts mouse-cortex
connectivity as well as a rule learned from the mouse itself** — the spatial wiring
principle appears universal across nematode and mammal.

## Organisms / data

- *C. elegans* — Witvliet et al. 2021 (8 developmental brain connectomes)
- *Drosophila* larva — Winding et al. 2023
- *Mus musculus* visual cortex — MICrONS `minnie65_public` (CAVE)

## Repo structure

- `src/phase1*` — edge-wise generator (v1, v2 with node embeddings + reciprocity)
- `src/phase2*` — functional verification (linear, LIF, OpenWorm c302)
- `src/phase3*` — functional optimization (RL / Evolution Strategies)
- `src/phase5*–9*` — cross-species, generation, realism (mouse)
- `src/phase10*–19*` — learning, layers, instinct, weighted gen, cross-species transfer
- `src/phase20*` — large-scale streaming generation (up to 300M neurons)
- `src/verify.py` — pre-publication verification with controls
- `PAPER.md` — manuscript draft · `PLAN.md` — full research log

## Reproduce

```bash
# Python 3.12, PyTorch (CUDA). Data downloaded separately (see PAPER.md §2).
python src/load_connectome.py        # build C. elegans tensors
python src/phase1_v2.py              # train generator
python src/verify.py                 # verify core claims
```

## AI-assistance disclosure

This project was carried out with **substantial assistance from an AI system (Claude)**,
which implemented the code, ran the experiments, and drafted the manuscript. The human
author directed the research, made decisions, caught errors, and takes responsibility for
the content. Results are **exploratory** and **not independently peer-reviewed**.

## Citation

If this is useful, please cite the archived release (Zenodo DOI — to be added).

## License

MIT (code). See [`LICENSE`](LICENSE).
