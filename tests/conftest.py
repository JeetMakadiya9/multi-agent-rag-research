"""Shared test setup: the per-process caches start empty for every test, so no test sees another's pages or
judgments (webrag.researcher.DEFAULT_READ_CACHE, docqa.verifier_v2.DEFAULT_CACHE)."""
import pytest


@pytest.fixture(autouse=True)
def _empty_process_caches():
    from docqa import verifier_v2
    from webrag import researcher
    researcher.DEFAULT_READ_CACHE.clear()
    verifier_v2.DEFAULT_CACHE.clear()
    yield
