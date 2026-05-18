"""
core.data_layer.ingestion.sources — per-source manifest loaders.

Each loader returns an iterable of `IngestionRecord` from a snapshotted
JSON manifest. No network I/O — manifests are committed under
`core/data_layer/ingestion/snapshots/<source>/<ticker>.json` so every
run is reproducible.

Loaders are intentionally tiny — they map manifest JSON to typed records.
"""

from __future__ import annotations
