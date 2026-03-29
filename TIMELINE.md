# Parameter Golf Armenia — Competition Timeline

**Competition:** March 24 – April 30, 2026
**Armenia prize deadline:** May 1, 2026
**Compute budget:** 20 GPU-hours on H100 (YSU Slurm)

> **Compute math:**
> - 1 full competition run (10 min × 8×H100) = 1.33 GPU-hours → ~15 runs total if all on 8×H100
> - Recommended split: ~12h on 1×H100 for dev + ~8h on 8×H100 (~6 full runs) for validation & final

---

## Hard Deadlines

| Date | Event |
|---|---|
| **April 13** | Submit to intermediate assessment (buffer day before deadline) |
| **April 14** | Optional intermediate assessment submission deadline |
| **April 18** | Intermediate results released |
| **April 30** | OpenAI competition closes |
| **May 1** | Armenia prize submission deadline |

---

## Phase 1 — Research & Setup · Mar 29–Apr 2 (4 days)

- [ ] Read all 25 submissions in `records/track_10min_16mb/` — note what techniques moved BPB the most
- [ ] Study the top 5 entries in detail (architecture changes, LR schedules, quantization tricks)
- [ ] Set up single-GPU test loop: short runs (~2000 iterations) for fast BPB signal
- [ ] Register for YSU compute if not done
- [ ] List 3–5 ideas to explore (with hypothesis for why each should help)

**Goal:** Enter Phase 2 with a shortlist of concrete, testable ideas.

---

## Phase 2 — Idea Exploration · Apr 2–Apr 9 (7 days)

- [ ] Run each idea on 1×H100 with reduced iterations for quick comparison
- [ ] Log every run: idea, hyperparams, BPB, GPU-hours used
- [ ] Focus budget on the most promising directions — don't over-iterate on dead ends
- [ ] Narrow down to 1–2 best approaches

**GPU budget:** ~12 GPU-hours on 1×H100
**Goal:** Identify the approach most likely to beat baseline on 8×H100.

---

## Phase 3 — First 8×H100 Validation · Apr 9–Apr 13 (4 days) · deadline Apr 13

- [ ] Run best idea as a full 10-min competition run on 8×H100
- [ ] Run at least 2 seeds to get variance estimate
- [ ] Prepare `submission.json`, `README.md`, `train.log`
- [ ] Submit to OpenAI leaderboard if score is competitive
- [ ] Submit to intermediate assessment by **April 13**

**GPU budget:** ~4 GPU-hours on 8×H100 (~3 full runs)
**Goal:** Confirm 8×H100 BPB, get intermediate feedback.

---

## Phase 4 — Feedback & Refinement · Apr 14–Apr 22 (8 days) · deadline Apr 14, results Apr 18

- [ ] Review intermediate results (released **April 18**)
- [ ] Note multilingual dataset performance — adjust if needed
- [ ] Continue single-GPU experiments on remaining ideas if budget allows
- [ ] Plan 1–2 final directions for Phase 5

**GPU budget:** ~4 GPU-hours on 1×H100 (if remaining)
**Goal:** Incorporate feedback, finalize best approach.

---

## Phase 5 — Final 8×H100 Runs · Apr 22–Apr 28 (6 days) · deadline Apr 30

- [ ] Run polished final version on 8×H100
- [ ] Run multiple seeds (statistical significance: p < 0.01 required for OpenAI leaderboard)
- [ ] Score must beat current SOTA by ≥ 0.005 nats BPB for leaderboard acceptance
- [ ] Pick best-scoring valid run

**GPU budget:** ~4 GPU-hours on 8×H100 (~3 full runs)
**Goal:** Best possible BPB within artifact constraints.

---

## Phase 6 — Final Submission · Apr 28–May 1 (3 days) · deadline May 1

- [ ] Create submission folder: `records/track_10min_16mb/<your_submission_name>/`
  - `train_gpt.py` — modified training script
  - `submission.json` — `{author, score_bpb, artifact_bytes, ...}`
  - `README.md` — approach description
  - `train.log` — full training output
- [ ] Verify artifact size ≤ 16MB: `len(train_gpt.py bytes) + zlib(int8_quantized_model)`
- [ ] Open PR to official OpenAI repo
- [ ] Submit to Armenia solution form by **May 1**

---

## GPU Budget Summary

| Phase | Mode | GPU-hours | Purpose |
|---|---|---|---|
| Phase 2 | 1×H100 | 12h | Rapid idea exploration |
| Phase 3 | 8×H100 | 4h | ~3 full validation runs |
| Phase 5 | 8×H100 | 4h | ~3 final competition runs |
| **Total** | | **20h** | |

---


## Ideas Backlog

Track ideas here as you research. Update with results.

| Idea | Hypothesis | Status | BPB (1×H100) | BPB (8×H100) |
|---|---|---|---|---|
| | | | | |

---
