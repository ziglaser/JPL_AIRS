"""Every quantity reported in Richardson et al. (2024) that our workflow can be
verified against, transcribed with its exact source in the paper.

``tests/test_paper_benchmarks.py`` turns each of these into an independent test
of the pipeline; notebooks import them for comparison tables. Keeping the numbers
here (not scattered through tests) means one file to re-check against the PDF.

Sources
-------
- "Methods"        : Methods section text (exact numbers).
- "Results"        : main-text Results (exact numbers).
- "Fig. 2c legend" : threshold values printed in the panel legend (exact).
- "Fig. 3 legend"  : per-hour Gini printed in the panel legends (exact, 2 d.p.).
- "Fig. 2b (read)" : values read off the plotted curves at 200 dpi -- digitisation
                     uncertainty ~ +/-0.02.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Sample definition
# --------------------------------------------------------------------------- #
#: Results: "The full sample size is N > 1 million".
MIN_TOTAL_SAMPLE: int = 1_000_000

#: Results: "(>160k per forecast hour)".
MIN_SAMPLE_PER_HOUR: int = 160_000

#: Results: "QPE99.95 corresponds to the top ~600 events" (=> N ~ 1.2 million).
APPROX_EVENTS_AT_9995: int = 600

#: Methods: "the latitude-longitude range is limited to 32-53N, 107-64W and grid
#: cells are excluded if their land fraction is below 50%."
DOMAIN_LAT: tuple[float, float] = (32.0, 53.0)
DOMAIN_LON: tuple[float, float] = (-107.0, -64.0)
LAND_FRACTION_MIN: float = 0.50

#: Methods: "AIRS retrievals are included during March-November 2019 and 2020".
MONTHS: tuple[int, int] = (3, 11)
YEARS: tuple[int, ...] = (2019, 2020)

#: Results: "approximately 13% of 1x1 grid cells do not return valid retrievals
#: during our study period."
AIRS_NONRETRIEVAL_FRACTION: float = 0.13

#: Methods (Gini): "the CDF shows an appropriate linear form at low values of
#: predictor, for example at CAPE below the 65th percentile" => ~65% of pooled
#: AIRS-FCST CAPE values are zero (the perturbed-tie mass).
ZERO_CAPE_FRACTION: float = 0.65

# --------------------------------------------------------------------------- #
# Pooled QPE percentile thresholds (mm/h)
# --------------------------------------------------------------------------- #
#: Fig. 2c legend (95th-99.9th) and Methods ("QPE > QPE99.95, which is
#: QPE > 5.1 mm h-1"). Thresholds are pooled over all locations/hours/wet+dry.
QPE_THRESHOLDS_MM_PER_H: dict[float, float] = {
    95.0: 0.1,
    99.0: 1.0,
    99.5: 1.7,
    99.9: 4.0,
    99.95: 5.1,
}

# --------------------------------------------------------------------------- #
# Fig. 2 statistics (pooled over the six forecast hours)
# --------------------------------------------------------------------------- #
#: Results: "In Fig. 2a, 80% of QPE > QPE99.95 events occur for CAPE > CAPE90,
#: meaning a probability of detection of 0.80."
POD_AT_CAPE90_QPE9995: float = 0.80

#: Results (Fig. 2c): "almost zero QPE > QPE99.9 events for CAPE < CAPE75".
#: We test the detection CDF at sample fraction 0.75 against this ceiling.
MAX_EVENT_FRACTION_BELOW_CAPE75_QPE999: float = 0.02

#: Fig. 2b (read): Gini vs QPE percentile, AIRS-FCST CAPE (red curve).
FIG2B_GINI_FCST: dict[float, float] = {
    95.0: 0.49,
    99.0: 0.70,
    99.5: 0.78,
    99.9: 0.86,
    99.95: 0.885,
}

#: Fig. 2b (read): Gini vs QPE percentile, AIRS overpass CAPE (blue curve).
FIG2B_GINI_OVERPASS: dict[float, float] = {
    95.0: 0.32,
    99.0: 0.46,
    99.5: 0.51,
    99.9: 0.60,
    99.95: 0.61,
}

#: Digitisation uncertainty for values read off Fig. 2b curves.
FIG2B_READ_TOL: float = 0.02

# --------------------------------------------------------------------------- #
# Fig. 3 statistics (per forecast hour, QPE99.95)
# --------------------------------------------------------------------------- #
#: Fig. 3a legend (exact): AIRS-FCST CAPE Gini by forecast hour UTC.
FIG3_GINI_FCST: dict[int, float] = {21: 0.78, 22: 0.84, 23: 0.86, 0: 0.86, 1: 0.89, 2: 0.87}

#: Fig. 3b legend (exact): AIRS overpass CAPE Gini by forecast hour UTC.
FIG3_GINI_OVERPASS: dict[int, float] = {21: 0.68, 22: 0.68, 23: 0.68, 0: 0.56, 1: 0.59, 2: 0.55}

#: Results/Methods: AIRS-FCST Gini "significantly improves with forecast hour
#: (p < 0.05)" and there are "significant declines in AIRS" -- via both the
#: bootstrapped hour-difference test and the OLS trend test.
FCST_TREND_SIGN: int = +1
OVERPASS_TREND_SIGN: int = -1

# --------------------------------------------------------------------------- #
# Uncertainty scale
# --------------------------------------------------------------------------- #
#: Methods: the p<0.05 hour-difference threshold 2*sqrt(2)*sigma "is approximately
#: a difference of +/-0.08 in Gini coefficient" => one-hour bootstrap sigma ~0.028.
#: Results quotes the looser "of order +/-0.01" for QPE99.95.
HOURLY_DIFF_P05: float = 0.08
HOURLY_BOOTSTRAP_SIGMA: float = 0.028

# =========================================================================== #
# Supplementary Information (docs/papers/Richardson_2024_Supplement.pdf)
# =========================================================================== #
#: Supp Notes 3: "with 455 days of valid data" (of the 550 possible Mar-Nov days).
VALID_DAYS: int = 455

#: Supp Notes 3: full-sample bootstrapped 95% CI on the QPE99.95 Gini,
#: "approximately [-0.018, 0.013]" => full-sample sigma of order 0.01.
FULL_SAMPLE_CI: tuple[float, float] = (-0.018, 0.013)

#: Supp Table 1 (exact): regional sub-sample sizes used for the threshold
#: sensitivity test. NOTE these imply NO land screen and near-complete
#: AIRS-FCST validity over the SE/Atlantic band incl. ocean (337,092 =
#: 99.6% of that band's cell-hours over 455 days), vs only ~28% over the
#: Plains -- the east-heavy validity gradient of parcel advection.
REGIONS: dict[str, dict] = {
    "Plains": dict(lat=(32.0, 53.0), lon=(-107.0, -96.0), n=176_352),
    "Midwest": dict(lat=(36.0, 53.0), lon=(-96.0, -80.0), n=384_822),
    "SE/Atlantic": dict(lat=(32.0, 36.0), lon=(-96.0, -65.0), n=337_092),
}

#: Supp Fig 13c legend (exact): the QPE-threshold ladder for QPE averaged over
#: the PRECIPITATING AREA of the cell only -- i.e. the semantics of our raw
#: ``MRMS_GaugeCorrQPE01H_av`` (kept as column ``qpe_wet``). Our data matches
#: all five to <2% -- the decisive confirmation of the _av semantics.
WET_QPE_THRESHOLDS_MM_PER_H: dict[float, float] = {
    95.0: 0.9,
    99.0: 3.1,
    99.5: 4.3,
    99.9: 7.1,
    99.95: 8.4,
}

#: Supp Fig 13b (read off curves, +/-0.02): Gini vs percentile against the
#: precipitating-area-mean QPE.
FIG13B_GINI_FCST: dict[float, float] = {
    95.0: 0.66, 99.0: 0.82, 99.5: 0.85, 99.9: 0.90, 99.95: 0.92,
}
FIG13B_GINI_OVERPASS: dict[float, float] = {
    95.0: 0.535, 99.0: 0.68, 99.5: 0.72, 99.9: 0.78, 99.95: 0.80,
}

#: Supp Fig 9a legend (exact): fraction of samples with CAPE = 0, conditioned
#: on QPE bins (keys: 'dry' = QPE=0, else QPE > QPE_X).
P_ZERO_CAPE: dict[str, float] = {
    "dry": 0.751, "99": 0.089, "99.9": 0.009, "99.95": 0.000,
}

#: Supp Fig 10a legend (exact): fraction of samples with QPE = 0, conditioned
#: on CAPE bins (keys: 'zero' = CAPE=0, else CAPE > CAPE_X).
P_DRY_GIVEN_CAPE: dict[str, float] = {
    "zero": 0.840, "99": 0.439, "99.9": 0.342, "99.95": 0.313,
}
