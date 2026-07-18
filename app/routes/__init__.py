"""Route modules registered by the composition root (``app.main.create_app``).

Each module exposes ``register(app, *, settings, runtime)`` and keeps its endpoints' closures
over the shared runtime/settings, exactly like the routes that remain in ``app.main``. The core
request/prompt/regression-image routes stay in ``app.main``; this package holds the surfaces
that grew past it (pdf benchmark, pdf document regression)."""
