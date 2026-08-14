"""Turn HYSPLIT forward trajectories into source-receptor soil-moisture influence
kernels.

See ``docs/TRAJECTORY_KERNEL_WORKPLAN.md`` for the design. The package
is built in the same style as :mod:`convection_skill`: a single cited
:mod:`~trajectory_kernels.config`, small pure functions, pluggable physics via
callables, and analytic unit tests.

The pipeline is a sequence of independently usable stages::

    trajectories -> pbl + contact -> resample + fuzz -> footprint -> io -> apply

The core scientific separation (forced by the data -- see the ``q`` note in
:mod:`~trajectory_kernels.trajectories`): the trajectory tool builds a purely
*geometric* residence-time footprint (where/when arriving air was in
boundary-layer contact with the land surface); soil-moisture physics enters only
by convolving that footprint with an external surface field
(:mod:`~trajectory_kernels.apply`).
"""

from __future__ import annotations
