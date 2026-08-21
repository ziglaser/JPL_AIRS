"""Static checks on the two dl_front chain scripts (C7/C8, 2026-08-18).

These scripts are submitted unattended overnight, so the cheapest possible
failure -- a shell syntax error, or a ``--help`` that stopped describing the
knobs an operator has to set -- must not survive to the cluster.  Nothing
here executes a phase; ``bash -n`` parses only, and ``--help`` is by
construction just the header comment block (both scripts implement the same
"help = everything above ``set -euo pipefail``" trick, so a code change can
never leak into the help text).

The two scripts deliberately duplicate each other's helpers rather than
sourcing a shared file -- the ablation experiments are kept SEPARATE from
the main curriculum chain (user decision 2026-08-18) so a bugfix in one can
never resubmit or retrain the other -- which is exactly why both are checked
here rather than one standing in for the other.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MAIN_CHAIN = REPO / "scripts/dlfront_jpl_chain.sh"
ABLATION_CHAIN = REPO / "scripts/dlfront_ablation_chain.sh"
FULL_SEQUENCE = REPO / "scripts/dlfront_full_sequence.sh"
ANALYSIS_CHAIN = REPO / "scripts/dlfront_analysis.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None,
                                reason="no bash on this machine")


def _help(script: Path) -> str:
    out = subprocess.run(["bash", str(script), "--help"], cwd=REPO, text=True,
                         capture_output=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return out.stdout


@pytest.mark.parametrize("script",
                         [MAIN_CHAIN, ABLATION_CHAIN, FULL_SEQUENCE,
                          ANALYSIS_CHAIN],
                         ids=lambda p: p.name)
def test_chain_script_parses(script):
    """``bash -n``: no syntax errors.  A chain that dies on load at 02:00
    wastes the whole overnight window."""
    assert script.exists()
    out = subprocess.run(["bash", "-n", str(script)], text=True,
                         capture_output=True, timeout=60)
    assert out.returncode == 0, out.stderr


@pytest.mark.parametrize("script", [MAIN_CHAIN, ABLATION_CHAIN,
                                    ANALYSIS_CHAIN],
                         ids=lambda p: p.name)
def test_help_is_the_header_comment(script):
    """``--help`` prints the header block: the scripts' only documentation.

    It must exit 0 (so ``script --help | less`` is usable), name the script,
    and stop before the code -- the sed trick that produces it slices up to
    ``set -euo pipefail``, so a leaked ``set -e`` line means the marker
    moved and the help is now printing shell.
    """
    text = _help(script)
    assert script.name in text
    assert len(text.splitlines()) > 20               # the real header
    assert "set -euo pipefail" not in text           # stops before the code
    for knob in ("JPL_AIRS_REPO", "JPL_AIRS_DATA", "JPL_AIRS_RESULTS",
                 "CLASSES", "FOLDS", "FORCE", "DRY_RUN"):
        assert knob in text, f"{script.name} --help no longer documents {knob}"


def test_main_chain_help_documents_stale_labels_and_force_eval():
    """C7: the two new behaviours must be discoverable from ``--help``.

    ``FORCE_EVAL=1`` is step 1 of the 2026-08-18 plan (re-score the existing
    checkpoints against the regenerated labels, no retraining); the
    labels_sha1 check is what makes that rerun happen automatically, since
    every _run.json written before today has no digest at all and therefore
    counts as stale.  Both are invisible in the code to an operator reading
    only the help.
    """
    text = _help(MAIN_CHAIN)
    assert "FORCE_EVAL" in text
    assert "labels_sha1" in text
    assert "label-digest" in text or "label digest" in text.lower()
    # graceful degradation is a promise to the operator, not an internal
    # detail: a submitting shell without the fronts-tf env must not abort
    assert "warning" in text.lower()


def test_ablation_chain_help_states_the_experiment_and_its_knobs():
    """C8: the ablation chain's header has to say WHY, near the top.

    An operator reading only ``--help`` must learn that U10M/V10M are WRF
    and SLP is MERRA-2, so the 2-channel rung is what AIRS actually
    supplies -- otherwise the ladder's numbers get read as a plain accuracy
    regression.  It must also document that the steps are independently
    runnable.

    And it must say that the 5-channel rung is RETRAINED as D6A5 rather
    than reusing the main chain's D6A (user decision 2026-08-18): D6A's
    weights predate the 2026-08-17 label fix, so a D6A-vs-D6A2 delta would
    conflate the channel effect with a label-quality effect -- and in the
    direction that flatters the low-channel rungs.  An operator who reads
    only --help must not be able to miss that.
    """
    text = _help(ABLATION_CHAIN)
    assert "WRF" in text and "MERRA-2" in text
    assert "T2M,QV2M" in text                         # the 2-channel rung
    assert "STEPS" in text                            # step selection knob
    assert "PERM_CKPTS" in text and "PERM_REPEATS" in text
    assert "CHANNEL_SETS" in text
    head = "\n".join(text.splitlines()[:40])
    assert "Step 2" in head and "Step 3" in head
    # the top rung is D6A5, trained fresh -- not the pre-fix D6A
    assert "D6A5" in text
    assert "D6A" in text and ("label" in text.lower() or "2026-08-17" in text)
