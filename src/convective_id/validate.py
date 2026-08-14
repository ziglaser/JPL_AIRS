"""Independent validation with AIRS-FCST CAPE (held out of every classifier).

The physical check: convection lives in high-CAPE environments. If a method's
"convective" cells are real convection, their MU CAPE distribution should sit
well above the non-convective precipitating cells' -- and, decisively, the
RESCUED cells (labeled convective despite a LOW PrecipFlag share: the
anvil/MCS cases the flags miss) should look like the flagged cores, not like
ordinary stratiform. Because no method ever saw CAPE (or any non-MRMS field),
this is an out-of-sample physical validation, not circular reasoning.

Metrics are computed on precipitating rows with finite CAPE (AIRS coverage
~59%); the AUC is the rank statistic P(CAPE_conv > CAPE_non) -- 0.5 = no
separation, 1.0 = perfect.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as sps  # noqa: E402

#: "Low flag share" bound defining the rescue set (well under the audit's
#: P99.9-event mean share of ~0.2 -- cells the flag-level filter would drop).
RESCUE_SHARE_MAX: float = 0.1


def cape_auc(cape_pos: np.ndarray, cape_neg: np.ndarray) -> float:
    """Rank AUC = P(CAPE_pos > CAPE_neg); NaN if either side is empty."""
    pos = cape_pos[np.isfinite(cape_pos)]
    neg = cape_neg[np.isfinite(cape_neg)]
    if pos.size == 0 or neg.size == 0:
        return np.nan
    u, _ = sps.mannwhitneyu(pos, neg, alternative="two-sided")
    return float(u / (pos.size * neg.size))


def _reference_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Flag-level reference classes, on the precip domain.

    Both references require pixel support (>=10 wet sub-pixels) so a 2-pixel
    shower cannot masquerade as a "core" (share 0.5 from 1 flagged pixel) or
    as clean stratiform.
    """
    precip = df["is_precip"].to_numpy()
    share = df["convective_share"].to_numpy(dtype=float)
    supported = df["wet_frac"].to_numpy() * 81.0 >= 10
    return {
        "flagged_core": precip & df["core_confident"].to_numpy(),
        "clean_stratiform": precip & supported & np.isfinite(share) & (share == 0)
                            & ~df["core_in_neighborhood"].to_numpy(),
    }


def validate_with_cape(
    df: pd.DataFrame,
    results: dict[str, pd.DataFrame],
    row_mask: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """One summary row per method (plus flag-level reference rows).

    Columns: sample sizes, the convective/non-convective CAPE medians and AUC,
    and the rescue-set diagnostics (labeled convective, flag share <
    RESCUE_SHARE_MAX): its CAPE median, AUC vs clean stratiform, and AUC vs
    the flagged cores (~0.5 = "rescued cells are CAPE-indistinguishable from
    cores", the success criterion).
    """
    keep = np.ones(len(df), dtype=bool) if row_mask is None else np.asarray(row_mask)
    precip = df["is_precip"].to_numpy() & keep
    cape = df["mu_cape"].to_numpy(dtype=float)
    share = df["convective_share"].to_numpy(dtype=float)
    refs = _reference_masks(df)
    core_cape = cape[refs["flagged_core"] & keep]
    strat_cape = cape[refs["clean_stratiform"] & keep]

    rows = []
    for name, res in results.items():
        conv = res["label"].to_numpy() & precip
        non = precip & ~conv
        rescued = conv & (~np.isfinite(share) | (share < RESCUE_SHARE_MAX))
        rows.append({
            "method": name,
            "n_convective": int(conv.sum()),
            "frac_of_precip": conv.sum() / max(precip.sum(), 1),
            "cape_median_conv": np.nanmedian(cape[conv]) if conv.any() else np.nan,
            "cape_median_non": np.nanmedian(cape[non]) if non.any() else np.nan,
            "cape_auc_conv_vs_non": cape_auc(cape[conv], cape[non]),
            "n_rescued": int(rescued.sum()),
            "rescued_frac_of_conv": rescued.sum() / max(conv.sum(), 1),
            "cape_median_rescued": (np.nanmedian(cape[rescued])
                                    if rescued.any() else np.nan),
            "rescued_auc_vs_stratiform": cape_auc(cape[rescued], strat_cape),
            "rescued_auc_vs_cores": cape_auc(cape[rescued], core_cape),
        })

    for ref_name, mask in refs.items():
        m = mask & keep
        rows.append({"method": f"[ref] {ref_name}", "n_convective": int(m.sum()),
                     "frac_of_precip": m.sum() / max(precip.sum(), 1),
                     "cape_median_conv": np.nanmedian(cape[m]) if m.any() else np.nan,
                     "cape_median_non": np.nan, "cape_auc_conv_vs_non": np.nan,
                     "n_rescued": 0, "rescued_frac_of_conv": np.nan,
                     "cape_median_rescued": np.nan,
                     "rescued_auc_vs_stratiform": np.nan,
                     "rescued_auc_vs_cores": np.nan})
    return pd.DataFrame(rows)


def agreement_matrix(results: dict[str, pd.DataFrame],
                     precip: np.ndarray) -> pd.DataFrame:
    """Pairwise Jaccard overlap of the convective labels on the precip domain."""
    names = list(results)
    out = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for i, a in enumerate(names):
        la = results[a]["label"].to_numpy() & precip
        for b in names[i + 1:]:
            lb = results[b]["label"].to_numpy() & precip
            union = (la | lb).sum()
            j = (la & lb).sum() / union if union else np.nan
            out.loc[a, b] = out.loc[b, a] = j
    return out


def plot_cape_validation(df: pd.DataFrame, results: dict[str, pd.DataFrame],
                         save_path: Path,
                         row_mask: Optional[np.ndarray] = None) -> None:
    """Per-method CAPE CDFs: convective / rescued / non-convective, against the
    flag-level core and clean-stratiform references (same in every panel)."""
    keep = np.ones(len(df), dtype=bool) if row_mask is None else np.asarray(row_mask)
    precip = df["is_precip"].to_numpy() & keep
    cape = df["mu_cape"].to_numpy(dtype=float)
    share = df["convective_share"].to_numpy(dtype=float)
    refs = _reference_masks(df)

    def cdf(ax, values, **kw):
        v = np.sort(values[np.isfinite(values)])
        if v.size:
            ax.plot(v, np.arange(1, v.size + 1) / v.size, **kw)

    fig, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results), 4),
                             sharey=True, sharex=True)
    for ax, (name, res) in zip(np.atleast_1d(axes), results.items()):
        conv = res["label"].to_numpy() & precip
        rescued = conv & (~np.isfinite(share) | (share < RESCUE_SHARE_MAX))
        cdf(ax, cape[refs["flagged_core"] & keep], color="0.2", lw=2.5,
            label="flagged cores (ref)")
        cdf(ax, cape[refs["clean_stratiform"] & keep], color="0.65", lw=2.5,
            label="clean stratiform (ref)")
        cdf(ax, cape[conv], color="crimson", lw=1.8, label="convective (method)")
        cdf(ax, cape[rescued], color="darkorange", lw=1.8, ls="--",
            label="rescued (low flag share)")
        cdf(ax, cape[precip & ~conv], color="steelblue", lw=1.8,
            label="non-convective")
        ax.set_xscale("symlog", linthresh=10)
        ax.set_xlabel("AIRS-FCST MU CAPE (J/kg)")
        ax.set_title(name)
    np.atleast_1d(axes)[0].set_ylabel("CDF")
    np.atleast_1d(axes)[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("CAPE validation (CAPE held out of every classifier)")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
