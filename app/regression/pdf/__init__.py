"""Document-level regression harness for ``translate_pdf`` (design doc §"Regression for
translate_pdf"). Per page the deterministic chain IS the image chain — the page fixtures reuse
``app.regression.pages.fixture`` verbatim; this package adds the document shell: census / raster /
text-layer-extraction checks on the frozen source PDF, assembly, and benchmark-on-replay."""
