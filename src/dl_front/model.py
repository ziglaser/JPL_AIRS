"""The DL-FRONT 2-D CNN (paper section 3.1, Fig. 1).

Layers 1-3: zero-pad -> 5x5 conv, 80 filters -> ReLU -> 50 % 2-D (spatial)
dropout.  Layer 4: zero-pad -> 5x5 conv, n_classes filters -> softmax.
'same' convolution == the paper's explicit zero-pad + valid 5x5 convolution.
Loss: per-category-weighted categorical cross-entropy (Eq. 4) with the
none-category weight 0.35, restricted to the Fig. 2 region mask via the
trailing y_true weight channel (house convention, front_finder.dataset).
"""
from __future__ import annotations

from . import config


def class_weight_vector(n_classes: int) -> list:
    """Section 3.1: none = 0.35, every front category (incl. dryline) = 1.0."""
    return [1.0] * (n_classes - 1) + [config.NONE_WEIGHT]


def make_loss(class_weights):
    """Weighted categorical cross-entropy over mask pixels (paper Eq. 4).

    y_true = one-hot (..., n_cls) + trailing pixel-weight channel; the loss
    is the weight-channel-weighted mean over pixels of
    -sum_c w_c * t_c * log(p_c).
    """
    import tensorflow as tf

    w = tf.constant(class_weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        onehot, pix_w = y_true[..., :-1], y_true[..., -1]
        p = tf.clip_by_value(y_pred, 1e-7, 1.0)
        ce = -tf.reduce_sum(onehot * tf.math.log(p) * w, axis=-1)
        return tf.reduce_sum(ce * pix_w) / tf.maximum(
            tf.reduce_sum(pix_w), 1.0)

    loss.__name__ = "weighted_cce"
    return loss


def masked_accuracy(y_true, y_pred):
    """Categorical accuracy over region-mask pixels (paper Table 2)."""
    import tensorflow as tf

    onehot, pix_w = y_true[..., :-1], y_true[..., -1]
    hit = tf.cast(tf.equal(tf.argmax(onehot, -1), tf.argmax(y_pred, -1)),
                  tf.float32)
    return tf.reduce_sum(hit * pix_w) / tf.maximum(tf.reduce_sum(pix_w), 1.0)


def build(n_classes: int, n_channels: int = len(config.SFC_VARS),
          learning_rate: float = config.LEARNING_RATE):
    """Compiled DL-FRONT CNN for (68, 141, n_channels) inputs.

    ~340k parameters at 5 classes -- the paper's exact architecture, so no
    size adaptation is needed for the GTX 1070.  ``n_channels`` grows by one
    when an input-validity channel is appended for the AIRS-degraded stages
    (replication runs use exactly the paper's 5).
    """
    import tensorflow as tf
    from tensorflow.keras import layers

    inp = layers.Input(shape=(None, None, n_channels))
    h = inp
    for i in range(config.N_CONV_LAYERS):
        h = layers.Conv2D(config.N_FILTERS, config.KERNEL_SIZE,
                          padding="same", activation="relu",
                          name=f"conv{i + 1}")(h)
        h = layers.SpatialDropout2D(config.DROPOUT, name=f"drop{i + 1}")(h)
    out = layers.Conv2D(n_classes, config.KERNEL_SIZE, padding="same",
                        activation="softmax",
                        name=f"conv{config.N_CONV_LAYERS + 1}")(h)
    model = tf.keras.Model(inp, out, name=f"dl_front_{n_classes}cls")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss=make_loss(class_weight_vector(n_classes)),
                  metrics=[masked_accuracy])
    return model
