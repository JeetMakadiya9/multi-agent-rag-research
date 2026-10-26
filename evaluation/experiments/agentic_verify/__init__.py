"""Agentic verification study (2026-10-22): a single-pass baseline and a bounded agentic workflow on SciFact.

New code, separate from the frozen Phase 10 systems and from the core-hypothesis harness. It executes the frozen
Phase 10 retriever (through ``core_hypothesis.smoke.build_live_retriever``) but does not import the frozen
Experiment B verifier: every arm here uses the verifier, planner and adjudicator prompts in ``prompts.py`` with the
same local model, so arms differ only in their control flow and evidence budget. See PROTOCOL.md.
"""
