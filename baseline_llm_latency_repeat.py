"""Concise repeat of the long unformatted verification latency case."""

from contextlib import redirect_stdout
import io
import json

from baseline_llm_latency_benchmark import (
    EXTRACTION_PROMPT,
    EXTRACTION_SYSTEM,
    MODEL,
    VERIFY_PROMPT,
    VERIFY_SYSTEM,
    call_benchmark,
)


keys = (
    "name",
    "model",
    "thinking_configuration",
    "temperature",
    "format_configuration",
    "prompt_characters",
    "start_utc",
    "end_utc",
    "time_to_first_content_seconds",
    "time_to_first_generated_content_seconds",
    "total_generation_seconds",
    "response_characters",
    "thinking_characters",
    "completed",
    "error",
    "ollama_final_event_metadata",
)


def concise_call(name, messages, *, think=False, output_format="json"):
    with redirect_stdout(io.StringIO()):
        result = call_benchmark(
            name,
            messages,
            think=think,
            output_format=output_format,
        )
    print(json.dumps({key: result[key] for key in keys}, ensure_ascii=False), flush=True)


concise_call(
    "extraction_think_false_json_format_repeat",
    [
        {"role": "system", "content": EXTRACTION_SYSTEM},
        {"role": "user", "content": EXTRACTION_PROMPT},
    ],
)
concise_call(
    "verification_think_false_json_format_repeat",
    [
        {"role": "system", "content": VERIFY_SYSTEM},
        {"role": "user", "content": VERIFY_PROMPT},
    ],
)
