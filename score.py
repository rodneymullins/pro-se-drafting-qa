#!/usr/bin/env python3
"""Score a pro se drafting task against its rubric.

Usage:
    python3 score.py <task-slug> <draft-file> [--judge-model MODEL]

Reads tasks/<slug>/task.json, extracts the draft text, and scores each
criterion. In --manual mode (default) it prints the criteria for a human
to self-assess. With --judge-model it calls an LLM judge via Ollama.

This is a QA loop, not legal advice. A licensed attorney reviews before filing.
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_task(slug: str) -> dict:
    path = ROOT / "tasks" / slug / "task.json"
    if not path.exists():
        raise FileNotFoundError(f"task.json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def extract_text(draft: Path) -> str:
    """Extract text from .docx, .md, or .txt."""
    suffix = draft.suffix.lower()
    if suffix == ".docx":
        from docx import Document
        doc = Document(str(draft))
        return "\n".join(p.text for p in doc.paragraphs)
    return draft.read_text(encoding="utf-8", errors="replace")


def manual_score(task: dict, draft_text: str) -> None:
    print(f"\n=== {task['title']} ===\n")
    print(f"Draft: {len(draft_text)} chars\n")
    passed = 0
    total = len(task["criteria"])
    for c in task["criteria"]:
        print(f"[ ] {c['id']}: {c['title']}")
        print(f"    {c['match_criteria'][:200]}")
        print()
    print(f"\n{passed}/{total} criteria passed (self-assess each above)")


def judge_score(task: dict, draft_text: str, model: str) -> None:
    """Score via an LLM judge (Ollama OpenAI-compatible endpoint)."""
    import openai
    client = openai.OpenAI(
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        api_key="ollama",
    )
    passed = 0
    total = len(task["criteria"])
    results = []
    for c in task["criteria"]:
        prompt = f"""You are a supervising attorney reviewing a pro se litigant's draft.

TASK: {task['title']}
INSTRUCTIONS: {task['instructions']}

CRITERION: {c['title']}
STANDARD: {c['match_criteria']}

DRAFT:
---
{draft_text[:8000]}
---

Respond with JSON: {{"verdict": "pass" or "fail", "reasoning": "one sentence"}}"""
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            text = resp.choices[0].message.content
            verdict = "pass" if '"pass"' in text.lower() else "fail"
            if verdict == "pass":
                passed += 1
            results.append((c["id"], c["title"], verdict))
            print(f"{'PASS' if verdict=='pass' else 'FAIL'}  {c['id']}: {c['title']}")
        except Exception as e:
            print(f"ERROR  {c['id']}: {e}")
            results.append((c["id"], c["title"], "error"))
    print(f"\n{passed}/{total} criteria passed")


def main():
    parser = argparse.ArgumentParser(description="Score a pro se drafting task")
    parser.add_argument("task", help="Task slug (e.g. surplus-funds-motion)")
    parser.add_argument("draft", help="Path to the draft file")
    parser.add_argument("--judge-model", default=None,
                        help="LLM judge model via Ollama (e.g. deepseek-v4-flash:cloud)")
    args = parser.parse_args()

    task = load_task(args.task)
    draft = Path(args.draft)
    if not draft.exists():
        raise FileNotFoundError(f"draft not found: {draft}")
    text = extract_text(draft)

    if args.judge_model:
        judge_score(task, text, args.judge_model)
    else:
        manual_score(task, text)


if __name__ == "__main__":
    main()
