"""Deterministic regression of the align + render stages.

The pipeline's non-deterministic stages (the grouping VLM, the translator) are *frozen* into a
**fixture** — OCR cells, the raw VLM hint string, and the per-unit translation payload — so a
replay re-runs only the deterministic chain (parse hint -> align -> render) and its output is
stable. The approved expected result is a **snapshot**: the align structure (units + ignored
cells) and the re-OCR of the rendered image. A regression run replays each fixture and diffs:
align exactly, render behaviourally via re-OCR (text + position).

The package mirrors the two harnesses: ``pages`` holds the shared per-page machinery
(fixture/snapshot model, replay, compare, re-OCR), ``image`` the image-fixture flow that
consumes it directly, and ``pdf`` the document harness that reuses it per page.

See ``docs/regression-test-design.md`` for the full design.
"""
from app.regression.pages.fixture import Fixture
from app.regression.pages.fixture import Snapshot
from app.regression.pages.replay import replay_fixture

__all__ = ["Fixture", "Snapshot", "replay_fixture"]
