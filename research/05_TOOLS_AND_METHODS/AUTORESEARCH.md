# Karpathy's Autoresearch

**Date**: 2026-03-21
**Source**: [github.com/karpathy/autoresearch](https://github.com/karpathy/autoresearch)
**Stars**: ~47.9k | **License**: MIT

---

## What Is It?

Autoresearch is an open-source project by Andrej Karpathy that sets up an **autonomous AI agent loop for ML research**. The agent is given a small but real LLM training setup on a single GPU and experiments autonomously — modifying code, training for exactly 5 minutes, checking if the result improved, keeping or discarding the change, and repeating indefinitely.

### Architecture: 3 Files

The entire system is remarkably simple:

| File | Role | Modifiable? |
|------|------|-------------|
| `prepare.py` | Downloads data, trains tokenizer, provides `evaluate_bpb()` and `make_dataloader()` | Read-only |
| `train.py` (~630 lines) | Complete GPT transformer with RoPE, flash attention, sliding window, hybrid MuonAdamW optimizer | **Agent modifies this** |
| `program.md` | Prompt/instructions for the AI agent — defines rules, constraints, loop behavior | Human-editable |

### The Autonomous Loop

1. Agent reads all three files, initializes a git branch (`autoresearch/<tag>`)
2. Agent modifies `train.py` with an experimental idea
3. Commits the change
4. Runs `uv run train.py > run.log 2>&1` (exactly 5 minutes wall-clock)
5. Extracts `val_bpb` (validation bits-per-byte) and `peak_vram_mb` from the log
6. Logs results to `results.tsv`
7. **If improved**: keeps the commit. **If not**: `git reset`
8. Repeats indefinitely without asking for human permission

### Key Constraints

- **Fixed 5-minute training budget** per experiment (~12 experiments/hour, ~100 overnight)
- **Single optimization metric**: `val_bpb` (lower is better)
- **Only `train.py` is modifiable** — no new package installs allowed
- **Simplicity preferred** — code deletions that maintain performance are valued
- Requires a single NVIDIA GPU (tested on H100)

### Technical Details of train.py

- GPT model with configurable depth/width
- Multi-head causal self-attention with flash-attention-3
- Rotary positional embeddings (RoPE)
- Sliding window attention patterns ("SSSL" for short/long)
- Hybrid optimizer: AdamW for embeddings + Muon for 2D matrix parameters
- LR scheduling tied to wall-clock time budget (warmup, constant, cooldown)
- Gradient accumulation to reach ~524K tokens per optimizer step

---

## Typical Use Cases

1. **Overnight autonomous hyperparameter/architecture search** — Let an agent run 100+ experiments while you sleep
2. **LLM training optimization** — Finding subtle improvements in transformer training (architecture tweaks, optimizer settings, learning rate schedules)
3. **Benchmarking AI agent capabilities** — Testing how well AI agents can do ML research autonomously
4. **Educational/research tool** — Understanding what modifications improve small LLM training

### Real-World Results

| Who | Experiments | Duration | Result |
|-----|-------------|----------|--------|
| Karpathy | 700 | 2 days | 20 optimizations found, 11% speedup on already-optimized code |
| Tobi Lutke (Shopify CEO) | 37 | overnight | 19% performance gain, 0.8B model outscored his previous 1.6B |
| SkyPilot (scaled) | ~910 | 8 hours | val_bpb from 1.003 to 0.974 |

Notable: the agent found a **bug Karpathy himself missed** — QK-Norm was missing a scalar multiplier.

### Karpathy's Vision

The future direction is making this "asynchronously massively collaborative for agents" — a SETI@home-style distributed research community of AI agents, not emulating a single PhD student but an **entire research community**.

---

## Is It a Match with XGBoost / Windmojo?

### The repo itself: NO

- **Hardcoded for LLM training** — the three files are specifically about training a GPT transformer on text data with PyTorch
- **Metric is `val_bpb`** (bits per byte) — not applicable to wind prediction
- **Requires NVIDIA GPU** with CUDA — XGBoost CPU training doesn't need this
- **5-minute budget** designed for neural network training — XGBoost with 30-50 features and ~2,300 samples trains in **seconds**

### The PATTERN: YES, extremely well

The autoresearch loop pattern — agent modifies code, runs experiment, checks metric, keeps/discards, repeats — could be powerful for wind prediction.

#### Mapping to Windmojo

| Autoresearch (LLM) | Windmojo Equivalent |
|---|---|
| `train.py` (GPT model) | Training script with XGBoost, feature selection, hyperparameters |
| `prepare.py` (data + eval) | Fixed data loading + MAE evaluation on held-out test weeks |
| `val_bpb` metric | `val_mae` (MAE on held-out weeks, currently 3.388 m/s) |
| 5-minute budget | **30-second budget** (XGBoost trains fast) |
| ~100 experiments overnight | **~1,000+ experiments overnight** (order of magnitude more) |
| Architecture search | Feature selection from 498 features, hyperparameter tuning |

#### What the Agent Could Explore

- **Feature selection** from our 498 features (Phase 2 showed this matters enormously — 30 vs 50 vs 150 features)
- **XGBoost hyperparameters** (max_depth, learning_rate, n_estimators, subsample, colsample_bytree, min_child_weight, gamma, reg_alpha, reg_lambda)
- **New engineered features** (interaction terms, different rolling windows, novel physics combinations)
- **Training data weighting** (seasonal weighting, afternoon emphasis for thermals)
- **Ensemble configurations** (blending multiple models)

#### What We'd Need to Build

1. **`prepare.py`** — Fixed script that loads Parquet features, provides `evaluate_mae()` on held-out test weeks (our 3 validation weeks)
2. **`train.py`** — Modifiable XGBoost training with feature selection and hyperparameters
3. **`program.md`** — Adapted for wind prediction:
   - Minimize MAE instead of val_bpb
   - Domain constraints (mandatory pressure gradient features, physics features)
   - Rules about feature categories (never drop all gradients, always keep cyclical encoding)
4. **The autonomous loop** — Git-based keep/discard logic (this part is model-agnostic)

#### Why This Could Be Even Better Than for LLMs

- **XGBoost trains in seconds** → 10-100x more experiments per night
- **Feature selection is a combinatorial problem** → perfect for autonomous search
- **Our Phase 2 already proved** that feature selection matters more than model complexity (30 features beat 150!)
- **Clear metric** (MAE on held-out weeks) with known baseline (GFS: 3.719 m/s)

---

## Conclusion

Autoresearch is not a drop-in tool for XGBoost work, but the **pattern it establishes** — autonomous agent loop with git-based experiment tracking and a single optimization metric — is highly applicable to Windmojo. The fast training time of XGBoost makes this pattern potentially even more powerful for our use case than for LLMs.

**Recommendation**: Consider building a Windmojo-specific autoresearch loop as a Phase 4 initiative. Expected effort: 1-2 days to set up the three files and loop logic. Expected payoff: autonomous discovery of feature/hyperparameter combinations we haven't tried.

---

## References

- [GitHub: karpathy/autoresearch](https://github.com/karpathy/autoresearch)
- [Fortune: 'The Karpathy Loop' — 700 experiments, 2 days](https://fortune.com/2026/03/17/andrej-karpathy-loop-autonomous-ai-agents-future/)
- [VentureBeat: Karpathy's autoresearch lets you run hundreds of AI experiments](https://venturebeat.com/technology/andrej-karpathys-new-open-source-autoresearch-lets-you-run-hundreds-of-ai)
- [SkyPilot Blog: Scaling Karpathy's Autoresearch](https://blog.skypilot.co/scaling-autoresearch/)
