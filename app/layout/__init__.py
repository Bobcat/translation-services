"""Document-layout component: the PP-DocLayout detector and the evidence read from it.

- ``detector``  the model engine and ``detect_layout_regions`` (plus the region label sets
                that classify its output);
- ``evidence``  what the pipeline concludes from regions over OCR cells: the document gate,
                column clustering, and preserve routing.

Shared by two consumers with different needs: the translation pipeline (align reads the
evidence on the ORIGINAL page) and the benchmark measurement layer (which detects on both
renders of a pair, at its own threshold). Public names re-exported here so callers depend on
the component, not its file layout.
"""
from app.layout.detector import PRESERVE_LABELS
from app.layout.detector import STRUCTURE_LABELS
from app.layout.detector import COLUMN_LABELS
from app.layout.detector import TEXT_LABELS
from app.layout.detector import detect_layout_regions
from app.layout.evidence import cell_columns
from app.layout.evidence import document_gate
from app.layout.evidence import preserved_cell_indices

__all__ = [
    "PRESERVE_LABELS",
    "STRUCTURE_LABELS",
    "COLUMN_LABELS",
    "TEXT_LABELS",
    "detect_layout_regions",
    "cell_columns",
    "document_gate",
    "preserved_cell_indices",
]
