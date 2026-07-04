# Complementary Learning Systems: Hippocampal-Cortical Memory Consolidation

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.19%2B-013243.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.3%2B-11557c.svg)](https://matplotlib.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A computational model of how the brain remembers: a fast-learning "hippocampus" and a slow-learning "cortex" work together so new memories form instantly without erasing old ones — based on McClelland, McNaughton & O'Reilly's 1995 CLS theory.

---

## What it does

- Implements two neural networks with very different learning rates — one fast (hippocampus), one slow (cortex)
- Simulates memory consolidation via **replay**: the hippocampus "teaches" the cortex offline, like during sleep
- Runs 5 experiments testing classic predictions of memory science
- Demonstrates and solves **catastrophic forgetting**, a core problem in both neuroscience and AI
- Simulates hippocampal lesions to reproduce real patterns of amnesia

---

## Background

**Complementary Learning Systems (CLS) theory** proposes the brain avoids a tradeoff between learning fast and remembering long-term by splitting the job:

| System | Learning | Representation | Role |
|---|---|---|---|
| **Hippocampus** | Fast | Sparse, pattern-separated | Quick episodic storage |
| **Neocortex** | Slow | Distributed, overlapping | Stable, long-term semantic memory |

**Consolidation** is the bridge between them: memories start out hippocampus-dependent, then get gradually transferred to cortex through repeated replay — explaining why damage to the hippocampus wipes out recent memories far more than old ones (temporally graded amnesia).

---

## Installation

```bash
git clone https://github.com/yourusername/complementary-learning-systems.git
cd complementary-learning-systems
python -m venv env && source env/bin/activate   # Windows: env\Scripts\activate
pip install -r requirements.txt
```

**requirements.txt**
```text
numpy>=1.19.0
matplotlib>=3.3.0
seaborn>=0.11.0
tqdm>=4.50.0
```

---

## Usage

```bash
python complementary_learning_systems.py
```

Runs all 5 experiments sequentially (~5–6 minutes total) and generates a figure for each.

### Customization

```python
cls = ComplementaryLearningSystem(
    input_size=20, hidden_size=50, output_size=10,
    hippo_lr=0.1,     # fast hippocampal learning
    cortex_lr=0.01    # 10x slower cortical learning
)

cls.replay_consolidation(n_replay_samples=15, n_iterations=10)
hippocampus.max_memory_size = 100   # episodic buffer capacity
```

---

## Experiments

| # | Question | Key Prediction | Output |
|---|---|---|---|
| 1 | Do the two systems learn at different speeds? | Hippocampus converges ~10x faster | `cls_learning_curves.png` |
| 2 | Does replay prevent forgetting? | Without replay, old memories are wiped out | `cls_consolidation_forgetting.png` |
| 3 | Does memory similarity drive interference? | More overlap → more interference; replay helps | `cls_interference.png` |
| 4 | When is the hippocampus critical? | Early lesions hurt more than late ones (graded amnesia) | `cls_hippocampal_lesion.png` |
| 5 | How does it all play out over time? | Cortex gradually catches up to hippocampus | `cls_comprehensive_analysis.png` |

---

## Results

| Finding | Result |
|---|---|
| Hippocampal vs. cortical learning speed | ~10x faster, ~6.7x lower final loss |
| Old memory loss, no replay vs. with replay | 0.39 → 0.04 (~9.8x better retention) |
| Interference reduction from replay | 60–80% |
| Cortical damage from Day 1 vs. Day 30 lesion | 5.2x worse vs. 1.1x worse than control |
| Replay events over 30-day simulation | 5,000+ |

**Takeaway:** fast hippocampal learning + slow cortical learning + offline replay reproduces real memory phenomena — rapid learning, resistance to interference, and graded retrograde amnesia — in a single simple model.

---

## Math, briefly

**Forward pass:** `h = σ(W₁ᵀx + b₁)`, `y = σ(W₂ᵀh + b₂)`

**Learning rule:** `W ← W − η∇L(W)`, with `η = 0.1` (hippocampus) or `0.01` (cortex)

**Replay consolidation:** repeatedly sample memories from the hippocampal buffer and update cortical weights on them

**Interference:** `I = L_after − L_before` (loss on old memories, before vs. after learning new ones)

**Memory similarity:** `X_new = α·X_old + (1−α)·noise`

---

## Roadmap

- Multiple consolidation timescales
- Recurrent connections for sequences
- Attention-based selective consolidation
- Hierarchical, multi-layer cortex
- Real-world (vision/language) tasks instead of synthetic patterns

---

## Contributing

Issues and PRs welcome — please follow PEP 8, add docstrings, and briefly explain the scientific motivation behind new features.

## License

MIT — see [LICENSE](LICENSE).

## References

- McClelland, McNaughton & O'Reilly (1995) — *Why there are complementary learning systems...*, Psychological Review
- Kumaran, Hassabis & McClelland (2016) — *What learning systems do intelligent agents need?*, Trends in Cognitive Sciences
- Frankland & Bontempi (2005) — *The organization of recent and remote memories*, Nature Reviews Neuroscience
- French (1999) — *Catastrophic forgetting in connectionist networks*, Trends in Cognitive Sciences

---

<div align="center">

**Understanding memory through computational modeling of brain systems**

</div>
