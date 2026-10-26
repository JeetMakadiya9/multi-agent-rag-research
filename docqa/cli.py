"""Command-line interface.

    python -m docqa --docs data/documents                      # interactive questions
    python -m docqa --docs data/documents --question "..."     # one question
    python -m docqa --docs data/documents --claim "..."        # verify one claim
"""
from __future__ import annotations

import argparse
import sys
from typing import Callable, List, Optional


def build_assistant(docs: str, model: str = "qwen3:4b", k: int = 4, verifier_name: str = "frozen_b",
                    log: Callable[[str], None] = print):
    """Load and index the documents and wire the pipeline. Returns (assistant, ingest report)."""
    from .answer import AnswerGenerator
    from .index import EvidenceIndex
    from .ingest import load_documents
    from .llm import OllamaChat
    from .pipeline import Assistant
    from .verify import VERIFIERS

    if verifier_name not in VERIFIERS:
        raise SystemExit(f"Unknown verifier {verifier_name!r}; choose from {sorted(VERIFIERS)}.")
    try:
        report = load_documents(docs)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from None
    for name, reason in report.skipped:
        log(f"[skipped] {name}: {reason}")
    if not report.pages:
        raise SystemExit(f"No readable TXT or PDF documents in {docs}.")
    log(f"[loaded] {len(report.documents)} document(s), {len(report.pages)} page(s): {', '.join(report.documents)}")
    index = EvidenceIndex(report.pages)
    client = OllamaChat(model=model)
    return Assistant(index, AnswerGenerator(client), VERIFIERS[verifier_name](client), k=k), report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m docqa", description="Evidence-grounded document QA.")
    parser.add_argument("--docs", default="data/documents", help="folder of .txt and .pdf files")
    parser.add_argument("--question", help="ask one question and exit")
    parser.add_argument("--claim", help="verify one claim and exit")
    parser.add_argument("--model", default="qwen3:4b", help="local Ollama model")
    parser.add_argument("--k", type=int, default=4, help="evidence passages per question")
    parser.add_argument("--verifier", default="frozen_b", help="claim verifier")
    args = parser.parse_args(argv)

    from .llm import LLMError
    from .pipeline import render_check, render_qa

    assistant, _ = build_assistant(args.docs, args.model, args.k, args.verifier)
    try:
        if args.claim:
            print(render_check(assistant.verify_claim(args.claim)))
            return 0
        if args.question:
            print(render_qa(assistant.ask(args.question)))
            return 0
        print("Type a question, 'check: <claim>' to verify a claim, or 'exit'.")
        while True:
            try:
                line = input("\n> ").strip()
            except EOFError:
                break
            if line.lower() in {"exit", "quit"}:
                break
            if line.lower().startswith("check:"):
                print(render_check(assistant.verify_claim(line[6:].strip())))
            elif line:
                print(render_qa(assistant.ask(line)))
    except LLMError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0
