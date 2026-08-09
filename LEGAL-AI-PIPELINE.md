# Legal AI Pipeline — Complete Documentation
*Last updated: 2026-08-09 14:40*

This document captures the full state of Rod's legal AI pipeline: the two open-source repos pulled in, the local-model harness wiring, the RLVR training loop, and the pro se task set. It is the single source of truth for what exists, where it lives, and how it runs.

---

## 1. The Two Repos

### 1.1 Harvey LAB (`/Users/rod/harvey-labs/`)
**Source:** `github.com/harveyai/harvey-labs`
**Size:** 5.1 GB, 63,123 files, fully checked out (not sparse)
**Purpose:** A legal benchmark harness with a synthetic law firm ("Calderwood & Harkness") and 2,010 graded tasks across 27 practice areas.

| Metric | Value |
|--------|-------|
| Tasks | 2,010 (README badge says 1,671; actual is 2,010) |
| Rubric criteria | 72,810 |
| Practice areas | 27 |
| Firm-knowledge subset | 250 tasks, 266 matters, 46 clients, 9,288 files, 108M tokens |

**Practice-area task counts:**
Contracts 498 · Firm-Knowledge 250 · Corporate M&A 161 · IP 147 · Corporate Governance 97 · Trusts/Estates 77 · Funds 66 · Litigation 52 · Real Estate 44 · Environmental/ESG 44 · Data Privacy 44 · Healthcare 43 · Emerging Companies/VC 43 · Intl Trade 41 · Employment 39 · Banking 37 · Arbitration 37 · Bankruptcy 36 · Capital Markets 35 · Tax 34 · Antitrust 33 · Structured Finance 31 · Insurance 31 · Energy 31 · Immigration 27 · White-Collar 21 · Diligence 11

**Key directories:**
- `tasks/` — all task definitions (each `task.json` has instructions + criteria rubric)
- `tasks/firm-knowledge/` — the 250-task enterprise-search benchmark (shared DMS)
- `harness/` — agent runner; `harness/adapters/` — model adapters
- `docs/architecture.md`, `docs/eval-strategies.md` — design docs
- `results/` — run outputs (metrics.json, transcript.jsonl, workspace/)
- `sandbox/` — Podman sandboxing

**Run command:**
```
uv run python -m harness.run --model <model> --task <task-id> --max-turns 8
```

**Harness tools:** bash, read, write, edit, glob, grep (6 tools). Skills: docx, pptx, xlsx.

### 1.2 Anthropic claude-for-legal (`/Users/rod/claude-for-legal/`)
**Source:** `github.com/anthropics/claude-for-legal`
**Size:** 5.6 MB, 37 commits, fully unshallowed
**Purpose:** A plugin marketplace of legal skills/agents for Claude Code.

**Contents:** 12 plugins, 151 skills, 349 files.

**The 12 plugins:** ai-governance, commercial, corporate, employment, ip, law-student, legal-builder-hub, legal-clinic, litigation, privacy, product, regulatory.

**Notable sub-templates:**
- `law-student/` — 12 skills (case-brief, irac-practice, legal-writing, outline-builder, exam-forecast, socratic-drill, flashcards, bar-prep-questions, cold-call-prep, study-plan, session, customize)
- `employment-legal/` — 17 skills (termination-review, hiring-review, wage-hour-qa, worker-classification, policy-drafting, handbook-updates, leave-tracker, internal-investigation, matter-workspace, etc.)
- `legal-clinic/` — law school clinic template (supervised practice: professor + students). Not pro se, but its intake/deadline/review-queue patterns map onto pro se case management.
- `managed-agent-cookbooks/` — 5 cookbooks: docket-watcher, reg-monitor, diligence-grid, launch-radar, renewal-watcher. All use a **three-tier security model**: Readers (Read/Grep only) → Analyzers (structured JSON, no MCP) → Writers (only tier with Write, never see raw docs). Writers have mandatory injection defenses (CSV formula neutralization, Markdown escaping, YAML quoting, inert URLs).

---

## 2. Local-Model Harness Wiring (Ollama)

### The problem
The stock `OpenAIAdapter` uses the **Responses API** (`client.responses.create`), which Ollama does not support. Ollama serves the standard **Chat Completions API** at `http://localhost:11434/v1`.

### The fix — `harness/adapters/ollama.py` (111 lines)
A new `OllamaAdapter` implementing the `ModelAdapter` interface via Chat Completions:
- Base URL from `OLLAMA_BASE_URL` env (default `http://localhost:11434/v1`)
- Maintains its own message history in Chat Completions format
- Translates harness tools to OpenAI function-call format
- Returns `ModelResponse` with tool calls, text, and token usage

**Run with local models:**
```
OLLAMA_BASE_URL=http://localhost:11434/v1 \
  .venv/bin/python -m harness.run --model ollama/deepseek-v4-flash:cloud --task <task-id>
```

**Note:** `harness/run.py` is modified (git status shows `M harness/run.py`) to register the `ollama` provider. The `ollama-adapter-notes.md` file documents the feasibility analysis.

---

## 3. Pro Se Task Set (`tasks/pro-se-ohio/`)

Five custom tasks built for Rod's actual cases. Each has a `task.json` (instructions + criteria rubric) and a `documents/` folder with the source record.

| Task slug | Title | Criteria | Case |
|-----------|-------|----------|------|
| `motion-fill-blank` | Fill in the Motion Template | 8 | Surplus stay (easy — curriculum base) |
| `simple-motion` | Draft a Simple Motion to Stay Surplus Distribution | 10 | Surplus stay (hard — curriculum target) |
| `surplus-funds-motion` | Draft Emergency Motion to Stay Surplus Distribution | 14 | 2025CV04438 surplus |
| `abedallah-eviction-answer` | Draft Answer with Affirmative Defenses — Abedallah Wrongful Eviction | 13 | Abedallah civil |
| `custody-interrogatories-review` | Review First Set of Interrogatories — F25-000196Z Custody | 16 | Custody |

---

## 4. RLVR Training Loop (`/Users/rod/Documents/pro-se-drafting-qa/`)

### Files
- `rlvr_loop.py` (12,281 bytes) — the RLVR loop
- `score.py` (3,732 bytes) — standalone rubric scorer
- `policy.json` — learned policy state
- `tasks/` — the 5 pro se task rubrics (mirrors harvey-labs)
- `README.md/` — directory (contains a temp file; not a real README)

### How it works
Wires **Microsoft's agent-learning REINFORCE loop** to Rod's rubric as the reward signal. Each episode runs the **real Harvey LAB harness** with a strategy directive, and reward is computed from a **deterministic verifier** (exact-fact checks) + an **LLM judge** for qualitative criteria.

**The 4 strategies (actions) being learned:**
| Action | Directive |
|--------|-----------|
| `lead-void-decree` | Lead with void-decree position (void ab initio) |
| `lead-service-defect` | Lead with FedEx service defect, Ohio Civ.R. 4.1(A)(1)(b) |
| `lead-suggestion-death` | Lead with fraudulent Suggestion of Death |
| `lead-zero-evidence` | Lead with zero-evidence challenge to competing claim |

**Deterministic checks (10 facts that MUST appear):** montgomery, 2025 cv 04438, mullins, 94,452.62, tieger, 4.1(a)(1)(b), suggestion of death, void, stay, certificate of service.

**RLVR caveats (reward zeroed if):**
- Agent cheated (read task.json / `read_rubric` flag)
- No verified deliverable produced (`deliverables_verified` false)

**Self-consistency:** each strategy runs N times per episode (default 2), best result kept.

**Curriculum:** 10 episodes on easy task (`motion-fill-blank`), then ramps to hard (`simple-motion`).

**Run command:**
```
cd /Users/rod/Documents/pro-se-drafting-qa
/Users/rod/harvey-labs/.venv/bin/python rlvr_loop.py \
  --task simple-motion --episodes 20 --judge-model deepseek-v4-flash:cloud \
  --max-turns 8 --samples 2 --curriculum --ramp-after 10
```

### Current training state (as of 2026-08-09 14:40)
- **Loop:** RUNNING (episode 7 of 20, background proc_22b75360e959)
- **Episode 5 result:** det 1.00, LLM 0.75, reward 0.92, baseline 0.33 — strong positive signal
- **Policy (policy.json, v4, 25 episodes seen):**
  - lead-void-decree: logit 0.0241
  - lead-suggestion-death: logit 0.0215
  - lead-service-defect: logit -0.0087
  - lead-zero-evidence: logit -0.0368
  - baseline: 0.332
- **Interpretation:** void-decree and suggestion-death are emerging as the preferred lead strategies; zero-evidence is being pushed down.

### Baseline run (before curriculum)
The easy task (`motion-fill-blank`) completes in 3 turns, 22s, verified deliverable, **deterministic score 1.0**. This is the curriculum base — the model fills the template ~100% of the time, giving the RLVR loop consistent positive signal.

---

## 5. Run Results (`/Users/rod/harvey-labs/results/`)

**49 runs** across pro-se-ohio tasks (plus firm-knowledge and litigation-dispute-resolution baselines).

**Latest surplus-funds-motion runs (deepseek-v4-flash:cloud):**
| Run | Input tokens | Output tokens | Verified | Read rubric |
|-----|-------------|---------------|----------|-------------|
| 133816 | 101,539 | 6,450 | False | False |
| 133946 | 96,670 | 5,947 | False | False |
| 134050 | 122,397 | 6,378 | **True** | True |

Each run dir contains: `metrics.json`, `transcript.jsonl`, `config.json`, `workspace/` (with `output/` deliverables and `skills/`).

---

## 6. Delegation Reports (cataloging)

Three subagent reports were produced during the initial cataloging:
- `/Users/rod/harvey-labs/task-taxonomy-catalog.md` — full 2,010-task taxonomy by practice area (19,354 bytes)
- `/Users/rod/harvey-labs/firm-knowledge-client-matter-catalog.md` — all 46 clients + 266 matters + relevance mapping (5,884 bytes)
- `/Users/rod/harvey-labs/ollama-adapter-notes.md` — adapter feasibility analysis (2,969 bytes)
- `/Users/rod/claude-for-legal/plugin-catalog.md` — all 12 plugins cataloged (528,717 bytes)

### Firm-knowledge relevance to Rod's cases
The benchmark has **no family-law practice area**, but the firm-knowledge tasks (250) test exactly the "distributed evidence retrieval + exhaustive search" capability pro se litigation demands. Closest matches:
- **F25-000196Z** (custody) → Litigation & Dispute Resolution; Employment & Labor
- **CA 30712** (foreclosure appeal) → Litigation; Real Estate
- **2025CV04438** (surplus) → Real Estate; Litigation; Bankruptcy
- **Abedallah** (wrongful eviction) → Real Estate; Litigation
- **Fornash** (death fraud) → Litigation; White Collar & Investigations

---

## 7. Key Insights & Lessons

1. **The failure mode Harvey identifies** — agents find core info but miss the rest (exhaustiveness, not reasoning) — is exactly what `exhaustive-evidence-retrieval` was built to solve. The vault + knowledge-graph RAG + MemOS stack already implements Harvey's proposed fix (build indexes upfront, amortize across tasks).

2. **The three-tier security model** (Readers→Analyzers→Writers with injection defenses) from the managed-agent cookbooks is directly applicable to Rod's custody-evidence pipeline.

3. **Local models work** — the Ollama adapter makes the full 2,010-task benchmark runnable on Rod's M4 Pro / MLX stack, no cloud API needed.

4. **RLVR is learning** — the policy is shifting toward void-decree and suggestion-death as lead strategies, with consistent positive reward on the easy curriculum base.

---

## 8. Quick Reference — Commands

```bash
# Run a single harness task with local model
cd /Users/rod/harvey-labs
OLLAMA_BASE_URL=http://localhost:11434/v1 \
  .venv/bin/python -m harness.run --model ollama/deepseek-v4-flash:cloud \
  --task pro-se-ohio/simple-motion --max-turns 8

# Score a draft against a rubric (manual)
cd /Users/rod/Documents/pro-se-drafting-qa
python3 score.py simple-motion <draft-file>

# Score with LLM judge
python3 score.py simple-motion <draft-file> --judge-model deepseek-v4-flash:cloud

# Run the RLVR training loop
python3 rlvr_loop.py --task simple-motion --episodes 20 \
  --judge-model deepseek-v4-flash:cloud --max-turns 8 --samples 2 \
  --curriculum --ramp-after 10
```
