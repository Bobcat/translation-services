"""The HTTP surface, one module per concern, registered by ``app.main.create_app``.

Each module exposes ``register(app, *, settings, runtime)`` and keeps its endpoints as
closures over the shared runtime/settings: ``requests`` (job submission + lifecycle),
``prompts`` (prompt library CRUD), ``image_regression`` and ``pdf_regression`` (the two
replay harnesses), ``benchmark`` (the document-pair benchmark), with the shared error
dialect and upload ceiling in ``common``."""
