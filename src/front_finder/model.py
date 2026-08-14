"""Build and compile the UNET3+ front detector (workplan section 3.5).

Thin wrapper over the vendored ``fronts/models/unets.py::unet_3plus``; the
only local logic is channel bookkeeping and the masked-FSS compile.
"""
from __future__ import annotations

import sys

from . import config, dataset

sys.path.insert(0, str(config.FRONTS_REPO))

#: Architecture constants (workplan 3.5; paper section 2a, rescaled for the
#: 1 deg grid, 2026-08-10).  The paper's 4-level / 5x5-kernel network was
#: sized for 0.25 deg: at 1 deg the same design has a ~120 deg receptive
#: field and an 8 deg/px bottleneck -- planetary scale, where the only
#: learnable structure is geography.  3 levels (2 poolings: 72x144 -> 18x36,
#: ~445 km/px bottleneck) + 3x3 kernels (~333 km/layer) land the receptive
#: field near the paper's IN KILOMETERS: synoptic scale, which is what
#: organizes fronts.  NOTE: pre-2026-08-10 checkpoints (levels=4, kernel 5)
#: cannot be --retrain'd into this architecture.
LEVELS = 3                        # 2 poolings: 72x144 -> 18x36 bottleneck
#: Paper per-tier widths for the 3 tiers kept; sized for one A100-80GB at
#: batch 64.  (The 2026-08-04 halved [16,32,64,128] local-GPU size is
#: obsolete -- pass --filter-num to shrink for small GPUs.)
FILTER_NUM = [32, 64, 128]
KERNEL_SIZE = 3
POOL_SIZE = (2, 2, 1)             # vertical dimension preserved (paper)
#: FSS neighborhood: 1 px ~ 111 km at 1 deg (paper used 1 px ~ 25 km at 0.25).
#: This is a UNIFORM 3x3 boxcar (AveragePooling), NOT distance-decaying:
#: flat tolerance within +-1 px, zero beyond (decision 2026-08-10; a
#: Gaussian-weighted pooling would decay but diverges from the paper).
FSS_MASK_SIZE = (3, 3)
FSS_ALPHA, FSS_BETA = 1.0, 0.5    # stock defaults in custom_losses.py


def make_loss(class_weights=None):
    """The masked FSS loss with this project's fixed hyperparameters.

    ``class_weights`` stays None in ALL training (decision 2026-08-10,
    resolving the workplan 3.5 contradiction in favor of the paper): the
    paper trains unweighted at worse imbalance than ours, the epoch-41
    quicklook showed no class collapse, and up-weighting rare classes
    pushes the frequency bias up -- see dataset.class_weights (diagnostics
    only) for the full rationale and the reopening criteria.
    """
    from models.custom_losses import masked_fractions_skill_score
    return masked_fractions_skill_score(
        mask_size=FSS_MASK_SIZE, alpha=FSS_ALPHA, beta=FSS_BETA,
        class_weights=class_weights)


def build(winds: bool, learning_rate: float = 1e-4, class_weights=None,
          filter_num=None):
    """Compiled unet_3plus for (72, 144, 5, C) inputs, n_cls classes.

    ``filter_num`` defaults to FILTER_NUM (A100 sizing); its length must
    equal LEVELS.
    """
    import tensorflow as tf
    from models.unets import unet_3plus

    n_ch = len(dataset.channel_names(winds))
    model = unet_3plus(
        input_shape=(*config.PADDED_SHAPE, len(config.TARGET_LEVELS_HPA), n_ch),
        num_classes=len(dataset.CLASS_NAMES),
        pool_size=POOL_SIZE, upsample_size=POOL_SIZE,
        levels=LEVELS, filter_num=list(filter_num or FILTER_NUM),
        kernel_size=KERNEL_SIZE,
        squeeze_axes=3, activation="gelu", batch_normalization=True,
        deep_supervision=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss=make_loss(class_weights))
    return model
