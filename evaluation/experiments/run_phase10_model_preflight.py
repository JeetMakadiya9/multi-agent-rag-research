"""Run the tiny local model readiness check without loading SciFact claims."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .phase10e_infrastructure import atomic_write_json, model_health_preflight


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--output", default="evaluation/experiments/results/phase10e_remediation/model_preflight.json")
    args = parser.parse_args()
    result = model_health_preflight(model=args.model, base_url=args.endpoint,
                                    timeout_seconds=args.timeout, temperature=0.0)
    result["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["provider"] = "OllamaCPUProvider-compatible local Ollama API"
    result["provider_configuration"] = {"base_url": args.endpoint.rstrip("/") + "/api/chat",
        "temperature": 0.0, "think": False, "num_gpu": 0, "num_predict": 16,
        "format": "json", "stream": False, "timeout_seconds": args.timeout}
    atomic_write_json(Path(args.output), result)
    print(json.dumps({"status": result["status"], "model": result["model"],
                      "model_digest": result["model_digest"],
                      "endpoint_reachable": result["endpoint_reachable"],
                      "json_capability": result["json_capability"],
                      "artifact": args.output, "errors": result["errors"]}, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
