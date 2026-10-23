"""docqa: evidence-grounded document question answering and claim verification.

A small, unfrozen application layer over the project's existing components:

* ``ingest``    - load local TXT and PDF files into page records (document name + page number);
* ``index``     - chunk and search them with the existing hybrid engine in ``rag.py`` (imported, never modified);
* ``llm``       - a local-only Ollama chat client that records truncation and timing;
* ``answer``    - answer generation in which every factual statement cites evidence passages;
* ``verify``    - claim verification (SUPPORTED / CONTRADICTED / INSUFFICIENT);
* ``pipeline``  - the end-to-end workflow used by the CLI and the evaluation harness;
* ``evaluate``  - a reproducible evaluation harness with explicit metrics.

Frozen research code (``rag.py``, ``experiment_b_verification.py``, ...) is imported read-only.
"""

# Opt-in conflict-adjudicated verifier modes: ``docqa.adjudicate`` adds them to ``docqa.verify.VERIFIERS`` when the
# package is first imported, so ``docqa/verify.py`` itself stays byte-identical to its pinned version (the heldout3
# protocol records its hash). Removing this import restores the previous registry exactly.
from . import adjudicate as _adjudicate  # noqa: E402,F401
