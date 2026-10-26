"""Command line:

    python -m webrag "research question"
    python -m webrag --verify "claim to check"
"""
from __future__ import annotations

import argparse
import sys

from .orchestrator import Orchestrator, Settings
from .report import render_markdown


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m webrag", description="Internet research and fact verification.")
    parser.add_argument("text", help="research question, or the claim when --verify is given")
    parser.add_argument("--verify", action="store_true", help="verify a claim instead of answering a question")
    parser.add_argument("--model", default="qwen3:4b")
    from docqa.verify import VERIFIERS
    parser.add_argument("--verifier", default="quote_stance", choices=sorted(VERIFIERS),
                        help="quote_stance (default, recommended); quote_stance_adjudicated (experimental opt-in: "
                             "adjudicates conflicting evidence; in exploratory runs it raised false support, see "
                             "evaluation/webrag/ADJUDICATED_VERIFIER.md); frozen_b; others are evaluation variants")
    parser.add_argument("--max-sources", type=int, default=10)
    parser.add_argument("--injection-guard", default="off", choices=("off", "flag", "exclude"),
                        help="screen retrieved passages for instruction-like text: off (default), flag or exclude")
    parser.add_argument("--quote-attestation", action="store_true",
                        help="record where each verifier quote occurs in its passage (record only: never changes a "
                             "verdict; needs a quote-based verifier)")
    args = parser.parse_args(argv)
    try:
        settings = Settings(model=args.model, verifier=args.verifier, max_sources=args.max_sources,
                            injection_guard=args.injection_guard, quote_attestation=args.quote_attestation)
    except ValueError as exc:                    # an invalid combination, e.g. --quote-attestation with frozen_b
        parser.error(str(exc))
    orchestrator = Orchestrator(settings)
    seen = set()

    def progress(data):
        for stage in data.get("stages", []):
            key = (stage["name"], stage["status"])
            if key not in seen:
                seen.add(key)
                print(f"[{stage['name']}] {stage['status']} ({stage['seconds']} s) {stage['detail']}", file=sys.stderr,
                      flush=True)

    data = orchestrator.run(args.text, "verify" if args.verify else "research", progress)
    print(render_markdown(data))
    return 0 if data["status"] != "FAILED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
