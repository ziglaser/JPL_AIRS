"""Minimal stand-in for the ``tensorflow`` module.

``fronts/utils/data_utils.py`` and ``fronts/utils/variables.py`` (vendored,
third-party) do an unconditional ``import tensorflow as tf`` at module scope,
and ``data_utils.expand_fronts`` even spells ``tf.Tensor`` in a bare (eagerly
evaluated) type annotation. That forces every consumer of
``front_finder.derive`` -- even pure-numpy callers who never pass a TF
tensor -- to have the real (huge) tensorflow package installed.

This stub exists ONLY so tests can import ``front_finder.derive`` on a
plain ``python3`` with no tensorflow installed. It supplies just enough
surface area (``Tensor``, ``is_tensor``, and the handful of ops referenced
inside ``if tf.is_tensor(...):`` branches) for the *numpy* code paths to run;
``is_tensor`` always returns False, so those TF-only branches are never
actually exercised by these tests. See test_front_dataset.py for the report
of this as a real bug in the vendored code.
"""
from __future__ import annotations

import numpy as _np


class Tensor:  # placeholder target for the `np.ndarray | tf.Tensor | ...` annotation
    pass


class Variable:
    def __init__(self, value):
        self._value = _np.asarray(value)

    def __getitem__(self, key):
        return self._value[key]

    def assign(self, value):
        self._value[...] = value


def is_tensor(x) -> bool:
    return False


def zeros_like(x):
    return _np.zeros_like(x)


def expand_dims(x, axis):
    return _np.expand_dims(x, axis)


def reduce_max(xs, axis=None):
    return _np.max(_np.stack(xs), axis=0 if axis is None else axis)


def where(cond, x, y):
    return _np.where(cond, x, y)


def pow(x, y):
    return _np.power(x, y)


def exp(x):
    return _np.exp(x)


def atan(x):
    return _np.arctan(x)


def sqrt(x):
    return _np.sqrt(x)


class math:
    exp = staticmethod(_np.exp)
    log = staticmethod(_np.log)


class data:
    class Dataset:
        @staticmethod
        def load(*a, **k):
            raise NotImplementedError("tensorflow stub: tf.data.Dataset.load unsupported")
