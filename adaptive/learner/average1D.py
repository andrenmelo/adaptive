# -*- coding: utf-8 -*-

from copy import deepcopy
import sys

import numpy as np

from adaptive.notebook_integration import ensure_holoviews
from adaptive.learner.learner1D import Learner1D
from adaptive.learner.average_mixin import add_average_mixin


@add_average_mixin
class AverageLearner1D(Learner1D):
    def __init__(self, function, bounds, loss_per_interval=None, weight=1):
        super().__init__(function, bounds, loss_per_interval)
        self._data = dict()  # {point: {seed: value}} mapping
        self.pending_points = dict()  # {point: {seed}}

        self.weight = weight
        self.min_values_per_point = 3

    def value_scale(self):
        return self._scale[1] or 1

    def _ask_points_without_adding(self, n):
        points, loss_improvements = super()._ask_points_without_adding(n)
        points = [(p, 0) for p in points]
        return points, loss_improvements

    def _get_neighbor_mapping_new_points(self, points):
        return {p: [n for n in self._find_neighbors(p, self.neighbors)
                    if n is not None] for p in points}

    def _get_neighbor_mapping_existing_points(self):
        return {k: [x for x in v if x is not None]
            for k, v in self.neighbors.items()}

    def unpack_point(self, x_seed):
        return x_seed

    def tell(self, x_seed, y):
        x, seed = x_seed

        in_data = x in self._data

        # either it is a float/int, if not, try casting to a np.array
        if not isinstance(y, (float, int)):
            y = np.asarray(y, dtype=float)

        self._add_to_data(x_seed, y)
        self._remove_from_to_pending(x_seed)
        self._update_data_structures(x, y)

    def tell_pending(self, x_seed):
        x, seed = x_seed

        self._add_to_pending(x_seed)

        if x not in self.neighbors_combined:
            # If 'x' already exists then there is not need to update.
            self._update_neighbors(x, self.neighbors_combined)
            self._update_losses(x, real=False)

    def tell_many(self, xs, ys):
        # `super().tell_many(xs, ys)` will not work.
        for x, y in zip(xs, ys):
            self.tell(x, y)

    def remove_unfinished(self):
        self.pending_points = {}
        self.losses_combined = deepcopy(self.losses)
        self.neighbors_combined = deepcopy(self.neighbors)

    def plot(self, *, with_sem=True):
        scatter = super().plot()
        if not with_sem:
            return scatter

        if self._data:
            hv = ensure_holoviews()
            xs, ys = zip(*sorted(self.data.items()))
            sem = self.data_sem
            err = [sem[x] if sem[x] < sys.float_info.max
                   else np.nan for x in xs]
            spread = hv.Spread((xs, ys, err))
            return scatter * spread
        else:
            return scatter
