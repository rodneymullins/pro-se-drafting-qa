#!/usr/bin/env python3
"""RLVR loop for pro-se drafting — learn which drafting strategy scores highest.

Wires Microsoft's agent-learning REINFORCE loop to OUR rubric as the
reward signal. Each episode runs the REAL Harvey LAB harness with a
strategy directive, and the reward is computed from a DETERMINISTIC
verifier (exact-fact checks) + an LLM judge for qualitative criteria.

The reward is zeroed if the agent cheated (read task.json) or produced no
verified deliverable — the RLVR caveat: reward the RESULT, not the FORM.

Self-consistency: each strategy is run N times per episode, and the best
result is kept, to reduce model variance.

Usage:
    python3 rlvr_loop.py --task simple-motion --episodes 20
"""

import argparse
import glob
import json
import os
import random
import subprocess
import sys
from pathlib import Path

# agent-learning SDK
sys.path.insert(0, "/Users/rod/agent-learning/src")
from agent_learning.policy.softmax_bandit import SoftmaxPolicy
from agent_learning.learners.reinforce import ReinforceLearner
from agent_learning.types import Action, Episode, Reward, RewardSource

QA_ROOT = Path("/Users/rod/Documents/pro-se-drafting-qa")
HARVEY_ROOT = Path("/Users/rod/harvey-labs")
POLICY_PATH = QA_ROOT / "policy.json"

# Strategy → prompt directive injected into the harness
STRATEGY_DIRECTIVES = {
    "lead-void-decree": "Lead the argument with the void-decree position (void ab initio due to service defect, fraudulent Suggestion of Death, and sale during stayed appeal).",
    "lead-service-defect": "Lead the argument with the FedEx service defect under Ohio Civ.R. 4.1(A)(1)(b) — service docketed without physical signed proof.",
    "lead-suggestion-death": "Lead the argument with the fraudulent Suggestion of Death filed by Fornash/Padgett claiming the party was deceased.",
    "lead-zero-evidence": "Lead the argument with the zero-evidence challenge to the competing claim (no note, no recorded mortgage, no ledger, no affidavit of debt).",
}

# Deterministic fact checks — the PRIMARY reward signal (RLVR verifiable reward)
# Each key is a fact that MUST appear in the draft. Exact, unambiguous.
DETERMINISTIC_CHECKS = {
    "court": ["montgomery"],
    "case_no": ["2025 cv 04438", "2025cv04438"],
    "party": ["mullins"],
    "surplus": ["94,452.62", "94452.62"],
    "tieger": ["tieger"],
    "fedex": ["4.1(a)(1)(b)", "4.1(a)(1)"],
    "suggestion_death": ["suggestion of death"],
    "void": ["void"],
    "stay": ["stay"],
    "cert_service": ["certificate of service", "served"],
}


def load_rubric(task_slug: str) -> dict:
    path = QA_ROOT / "tasks" / task_slug / "task.json"
    return json.loads(path.read_text(encoding="utf-8"))


def deterministic_verify(draft_text: str) -> float:
    """Deterministic fact-presence check. Returns pass-rate in [0,1].

    This is the RLVR verifiable reward — exact, unambiguous, no judge
    variance. A fact is either present or not.
    """
    if not draft_text:
        return 0.0
    text = draft_text.lower()
    passed = 0
    for key, variants in DETERMINISTIC_CHECKS.items():
        if any(v in text for v in variants):
            passed += 1
    return passed / len(DETERMINISTIC_CHECKS)


def llm_judge(draft_text: str, rubric: dict, judge_model: str) -> float:
    """LLM judge for QUALITATIVE criteria (argument quality, relief).

    Used as a secondary signal on top of the deterministic verifier.
    """
    import openai
    client = openai.OpenAI(
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        api_key="ollama",
    )
    passed = 0
    total = len(rubric["criteria"])
    for c in rubric["criteria"]:
        prompt = f"""You are a supervising attorney reviewing a pro se litigant's draft.

TASK: {rubric['title']}
CRITERION: {c['title']}
STANDARD: {c['match_criteria']}

DRAFT:
---
{draft_text[:8000]}
---

Respond with JSON: {{"verdict": "pass" or "fail"}}"""
        try:
            resp = client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            if '"pass"' in resp.choices[0].message.content.lower():
                passed += 1
        except Exception:
            pass
    return passed / total if total else 0.0


def run_agent_episode(strategy: str, task_slug: str, agent_model: str, max_turns: int) -> dict:
    """Run the real Harvey LAB harness once under a strategy."""
    directive = STRATEGY_DIRECTIVES.get(strategy, "")
    cmd = [
        str(HARVEY_ROOT / ".venv/bin/python"), "-m", "harness.run",
        "--model", f"ollama/{agent_model}",
        "--task", f"pro-se-ohio/{task_slug}",
        "--max-turns", str(max_turns),
        "--strategy", directive,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=900, cwd=str(HARVEY_ROOT)
        )
    except subprocess.TimeoutExpired:
        return {"draft": "", "verified": False, "read_rubric": False, "error": "timeout"}

    runs = sorted(glob.glob(
        f"{HARVEY_ROOT}/results/pro-se-ohio/{task_slug}/{agent_model}/*/"
    ))
    if not runs:
        return {"draft": "", "verified": False, "read_rubric": False, "error": "no run dir"}

    latest = runs[-1]
    draft = ""
    for f in glob.glob(f"{latest}/output/*"):
        if f.endswith((".docx", ".md", ".txt")):
            try:
                if f.endswith(".docx"):
                    from docx import Document
                    doc = Document(f)
                    draft = "\n".join(p.text for p in doc.paragraphs)
                else:
                    draft = Path(f).read_text(errors="replace")
            except Exception:
                draft = ""
            break

    metrics_path = Path(f"{latest}/metrics.json")
    verified = False
    read_rubric = False
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
            verified = metrics.get("deliverables_verified", False)
            read_rubric = metrics.get("read_rubric", False)
        except Exception:
            pass

    return {
        "draft": draft,
        "verified": verified,
        "read_rubric": read_rubric,
        "run_dir": str(latest),
    }


def save_policy(policy, path: Path) -> None:
    snap = policy.snapshot()
    path.write_text(json.dumps({
        "logits": snap.logits,
        "baseline": snap.baseline,
        "episodes_seen": snap.episodes_seen,
        "version": snap.version,
    }, indent=2))


def load_policy(policy, path: Path) -> None:
    if path.exists():
        data = json.loads(path.read_text())
        policy._snapshot.logits = data["logits"]
        policy._snapshot.baseline = data["baseline"]
        policy._snapshot.episodes_seen = data["episodes_seen"]
        policy._snapshot.version = data["version"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="simple-motion")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--judge-model", default="deepseek-v4-pro:cloud",
                        help="LLM judge model (different from agent)")
    parser.add_argument("--agent-model", default="deepseek-v4-flash:cloud",
                        help="Agent model for the harness (the drafter)")
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--samples", type=int, default=2,
                        help="Self-consistency: runs per strategy per episode")
    parser.add_argument("--llm-weight", type=float, default=0.3,
                        help="Weight of LLM judge vs deterministic verifier")
    parser.add_argument("--curriculum", action="store_true",
                        help="Start on easy task, ramp to harder after N episodes")
    parser.add_argument("--ramp-after", type=int, default=10,
                        help="Episodes on easy task before ramping to hard")
    args = parser.parse_args()

    rubric = load_rubric(args.task)
    print(f"Task: {rubric['title']}")
    print(f"Criteria: {len(rubric['criteria'])}")
    print(f"Deterministic checks: {len(DETERMINISTIC_CHECKS)}")
    print(f"Self-consistency: {args.samples} samples/strategy")
    if args.curriculum:
        print(f"Curriculum: {args.ramp_after} eps on easy, then ramp to hard")

    # Curriculum: easy task first, hard task after ramp
    EASY_TASK = "motion-fill-blank"
    HARD_TASK = "simple-motion"
    active_task = args.task
    if args.curriculum:
        active_task = EASY_TASK
        rubric = load_rubric(EASY_TASK)

    actions = [
        Action(id="lead-void-decree", description="Lead with void decree argument"),
        Action(id="lead-service-defect", description="Lead with service defect"),
        Action(id="lead-suggestion-death", description="Lead with Suggestion of Death"),
        Action(id="lead-zero-evidence", description="Lead with zero-evidence challenge"),
    ]

    policy = SoftmaxPolicy.from_actions(
        actions, agent_id="pro-se", task_id=args.task, rng=random.Random(args.seed)
    )
    load_policy(policy, POLICY_PATH)
    learner = ReinforceLearner()

    print("\n=== RLVR LOOP (deterministic reward) ===")
    print(f"{'Ep':<4} {'Action':<22} {'Det':<6} {'LLM':<6} {'Reward':<8} {'Base':<6} {'V':<5} {'C':<5} {'Probs'}")
    print("-" * 90)

    episodes = []
    rewards = []
    for ep in range(1, args.episodes + 1):
        # Curriculum ramp: switch to hard task after ramp_after episodes
        if args.curriculum and ep == args.ramp_after + 1:
            active_task = HARD_TASK
            rubric = load_rubric(HARD_TASK)
            print(f"\n  >>> RAMPING to hard task: {rubric['title']} <<<\n")

        chosen = policy.choose()
        strategy = chosen.action.id

        # Self-consistency: run the strategy N times, keep the best
        best_det = 0.0
        best_draft = ""
        best_flags = {"verified": False, "read_rubric": False}
        for s in range(args.samples):
            print(f"  [ep {ep}] {strategy} (sample {s+1}/{args.samples})...")
            res = run_agent_episode(strategy, active_task, args.agent_model, args.max_turns)
            det = deterministic_verify(res["draft"])
            if det > best_det:
                best_det = det
                best_draft = res["draft"]
                best_flags = {"verified": res["verified"], "read_rubric": res["read_rubric"]}

        # LLM judge for qualitative criteria (secondary signal)
        llm_score = llm_judge(best_draft, rubric, args.judge_model) if best_draft else 0.0

        # Combine: deterministic (primary) + LLM (secondary)
        reward_value = (1 - args.llm_weight) * best_det + args.llm_weight * llm_score

        # RLVR caveat: penalize cheating and false success
        if best_flags["read_rubric"]:
            reward_value = 0.0
        if not best_flags["verified"]:
            reward_value = 0.0

        episode = Episode(
            agent_id="pro-se",
            task_id=args.task,
            action_id=strategy,
            action_logprob=chosen.logprob,
            intent_summary="draft motion",
            expected_outcome="high rubric score",
            execution_status="completed",
            result_summary=f"det={best_det:.2f} llm={llm_score:.2f}",
        )
        reward = Reward(
            episode_id=episode.id,
            agent_id="pro-se",
            source=RewardSource.AGGREGATE,
            value=reward_value,
            rubric=args.task,
        )
        episodes.append(episode)
        rewards.append(reward)

        if ep % 5 == 0:
            result = learner.update(policy, episodes, rewards)
            probs = [f"{p:.2f}" for p in policy.probabilities()]
            print(f"{ep:<4} {strategy:<22} {best_det:<6.2f} {llm_score:<6.2f} "
                  f"{reward_value:<8.2f} {result.baseline_after:<6.2f} "
                  f"{str(best_flags['verified']):<5} {str(best_flags['read_rubric']):<5} {probs}")
            save_policy(policy, POLICY_PATH)

    print("\n=== FINAL POLICY ===")
    probs = policy.probabilities()
    for a, p in zip(actions, probs):
        print(f"  {a.id:<22} {p:.2%}")
    best = max(zip(actions, probs), key=lambda x: x[1])
    print(f"\nBest strategy: {best[0].id} ({best[1]:.1%})")
    save_policy(policy, POLICY_PATH)
    print(f"Policy saved to {POLICY_PATH}")


if __name__ == "__main__":
    main()
