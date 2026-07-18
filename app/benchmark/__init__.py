"""Document-pair benchmark for translate_pdf (docs/pdf-benchmark-regression-design.md).

- ``measurement``  the expensive, environment-bound layer: render both PDFs,
                   PP-DocLayout + OCR per page -> measurement dict (frozen per run)
- ``scoring``      pure functions: measurement -> per-axis scores + flags
- ``store``        persistent run storage under data/benchmark/ (outside work_root)
"""
