"""Deterministic regression of the align + render stages.

The pipeline's non-deterministic stages (the grouping VLM, the translator) are *frozen* into a
**fixture** — OCR cells, the raw VLM hint string, and the per-unit translation payload — so a
replay re-runs only the deterministic chain (parse hint -> align -> render) and its output is
stable. The approved expected result is a **snapshot**: the align structure (units + ignored
cells) and the re-OCR of the rendered image. A regression run replays each fixture and diffs:
align exactly, render behaviourally via re-OCR (text + position).

See ``docs/regression-test-design.md`` for the full design.
"""
from app.regression.fixture import Fixture
from app.regression.fixture import Snapshot
from app.regression.replay import replay_fixture

__all__ = ["Fixture", "Snapshot", "replay_fixture"]
