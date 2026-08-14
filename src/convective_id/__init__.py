"""Convective-storm identification from MRMS structure alone.

THE PROBLEM: the MRMS PrecipFlag "Convection" category marks strict convective
CORES. In a 1x1-deg cell containing a big convective storm, most of the rain
(anvil, MCS trailing stratiform) is flagged stratiform, so flag-share
thresholds systematically miss organized convection (audit: P99.9 QPE events
average convective share ~0.2; only ~2% are majority-convective).

THE APPROACH: classify precipitating cell-hours as convective using ONLY MRMS
data as input -- no AIRS or SMAP anywhere in the features, so the labels can
later be used alongside those products without double-dipping. Four methods,
one shared output contract (boolean label + continuous score per row):

- ``methods.threshold_classify``  : physical cuts on sub-pixel structure
  (peak rate, skewness, conditional intensity).
- ``methods.cluster_classify``    : unsupervised GMM in flag-free feature
  space; the "convective" component is identified by its MRMS profile
  (highest sub-pixel peak), never by CAPE.
- ``methods.forest_classify``     : weak supervision -- confident flag-level
  cores are positives, core-free surroundings negatives; the forest sees only
  flag-free structure, so it generalizes core-like STRUCTURE to anvil cells
  the flags miss.
- ``methods.object_classify``     : storm objects -- connected precipitating
  components per (date, slot); a component containing any core seed is
  convective THROUGHOUT (the anvil is attached to its storm; this is the
  most direct fix for the stated problem).

VALIDATION (``validate``): AIRS-FCST MU CAPE is held out of every method and
used only to check the classifications afterwards -- convective-labeled cells
should sit in higher-CAPE environments, and critically the RESCUED cells
(labeled convective despite a low flag share -- the anvil/MCS cases) should
look CAPE-wise like flagged cores, not like ordinary stratiform.

Run the whole thing: ``PYTHONPATH=src python -m convective_id.demo [year]``.
"""

__all__ = ["features", "methods", "plotting", "validate"]
