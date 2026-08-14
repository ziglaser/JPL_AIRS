"""Front & dryline detection from AIRS data (FrontFinder methodology).

See docs/FRONT_DETECTION_WORKPLAN.md. Stages: ingest -> derive -> regrid ->
dataset -> train -> evaluate; this package consumes the vendored ``fronts/``
codebase (Justin et al. 2025, AIES-D-24-0043) as a library.
"""
