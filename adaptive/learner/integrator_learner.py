# -*- coding: utf-8 -*-
# Copyright 2010 Pedro Gonnet
# Copyright 2017 Christoph Groth
# Copyright 2017 `adaptive` authors

from collections import defaultdict
from math import sqrt
from operator import attrgetter

import holoviews as hv
import numpy as np
from scipy.linalg import norm
from sortedcontainers import SortedDict, SortedSet

from .base_learner import BaseLearner
from .integrator_coeffs import (b_def, T_left, T_right, ns,
                                xi, V_inv, Vcond, alpha, gamma)


eps = np.spacing(1)


def _downdate(c, nans, depth):
    b = b_def[depth].copy()
    m = ns[depth] - 1
    for i in nans:
        b[m + 1] /= alpha[m]
        xii = xi[depth][i]
        b[m] = (b[m] + xii * b[m + 1]) / alpha[m - 1]
        for j in range(m - 1, 0, -1):
            b[j] = ((b[j] + xii * b[j + 1] - gamma[j + 1] * b[j + 2])
                    / alpha[j - 1])
        b = b[1:]

        c[:m] -= c[m] / b[m] * b[:m]
        c[m] = 0
        m -= 1


def _zero_nans(fx):
    """Caution: this function modifies fx."""
    nans = []
    for i in range(len(fx)):
        if not np.isfinite(fx[i]):
            nans.append(i)
            fx[i] = 0.0
    return nans


def _calc_coeffs(fx, depth):
    """Caution: this function modifies fx."""
    nans = _zero_nans(fx)
    c_new = V_inv[depth] @ fx
    if nans:
        fx[nans] = np.nan
        _downdate(c_new, nans, depth)
    return c_new


class DivergentIntegralError(ValueError):
    pass


class Interval:

    """
    Attributes
    ----------
    (a, b) : (float, float)
        The left and right boundary of the interval.
    c : numpy array of shape (4, 33)
        Coefficients of the fit.
    c_old : numpy array of shape `len(fx)`
        Coefficients of the fit.
    depth : int
        The level of refinement, `depth=0` means that it has 5 (the minimal number of) points and
        `depth=3` means it has 33 (the maximal number of) points.
    fx : numpy array of size `(5, 9, 17, 33)[self.depth]`.
        The function values at the points `self.points(self.depth)`.
    igral : float
        The integral value of the interval.
    err : float
        The error associated with the integral value.
    tol : float
        The relative tolerance that needs to be reached in the precision of the integral.
    rdepth : int
        The number of splits that the interval has gone through, starting at 1.
    ndiv : int
        A number that is used to determine whether the interval is divergent.
    parent : Interval
        The parent interval. If the interval resulted from a refinement, it has one parent. If
        it resulted from a split, it has two parents.
    children : list of `Interval`s
        The intervals resulting from a split or refinement.
    done_points : dict
        A dictionary with the x-values and y-values: `{x1: y1, x2: y2 ...}`.
    est_err : float
        The sum of the errors of the children, if one of the children is not ready yet,
        the error is infinity.
    discard : bool
        If True, the interval and it's children are not participating in the
        determination of the total integral anymore because its parent had a
        refinement when the data of the interval was not known, and later it appears
        that this interval has to be split.
    complete : bool
        All the function values in the interval are known. This does not necessarily mean
        that the integral value has been calculated, see `self.done`.
    done : bool
        The integral and the error for the interval has been calculated.
    branch_complete : bool
        The interval can be used to determine the total integral, however if its children are
        also `branch_complete`, they should be used.

    """

    __slots__ = ['a', 'b', 'c', 'c_old', 'depth', 'fx', 'igral', 'err', 'tol',
                 'rdepth', 'ndiv', 'parent', 'children', 'done_points',
                 'est_err', 'discard']

    def __init__(self, a, b):
        self.children = []
        self.done_points = SortedDict()
        self.a = a
        self.b = b
        self.c = np.zeros((len(ns), ns[-1]))
        self.est_err = np.inf
        self.discard = False
        self.igral = None

    @classmethod
    def make_first(cls, a, b, tol):
        ival = Interval(a, b)
        ival.tol = tol
        ival.ndiv = 0
        ival.rdepth = 1
        ival.parent = None
        ival.depth = 3
        ival.c_old = np.zeros(ns[ival.depth])
        ival.err = np.inf
        return ival, ival.points(ival.depth)

    @property
    def complete(self):
        """The interval has all the y-values to calculate the intergral."""
        return len(self.done_points) == ns[self.depth]

    @property
    def done(self):
        """The interval is complete and has the intergral calculated."""
        return hasattr(self, 'fx') and self.complete

    @property
    def branch_complete(self):
        if not self.children and self.complete:
            return True
        else:
            return np.isfinite(sum(i.est_err for i in self.children))

    @property
    def T(self):
        """Get the correct shift matrix.

        Should only be called on children of a split interval.
        """
        assert self.parent is not None
        left = self.a == self.parent.a
        right = self.b == self.parent.b
        assert left != right
        return T_left if left else T_right

    def points(self, depth):
        a = self.a
        b = self.b
        return (a+b)/2 + (b-a)*xi[depth]/2

    def refine(self):
        ival = Interval(self.a, self.b)
        ival.tol = self.tol
        ival.rdepth = self.rdepth
        ival.ndiv = self.ndiv
        ival.c = self.c.copy()
        ival.c_old = self.c_old.copy()
        ival.parent = self
        self.children = [ival]
        ival.err = self.err
        ival.depth = self.depth + 1
        points = ival.points(ival.depth)
        return ival, points

    def split(self):
        points = self.points(self.depth)

        a = self.a
        b = self.b
        m = points[len(points) // 2]

        ivals = [Interval(a, m), Interval(m, b)]
        self.children = ivals

        for ival in ivals:
            ival.depth = 0
            ival.tol = self.tol / sqrt(2)
            ival.c_old = self.c_old.copy()
            ival.rdepth = self.rdepth + 1
            ival.parent = self
            ival.ndiv = self.ndiv
            ival.err = self.err / sqrt(2)

        return ivals

    def complete_process(self):
        """Calculate the integral contribution and error from this interval,
        and update the estimated error of all ancestor intervals."""
        force_split = False
        if self.parent is None:
            self.process_make_first()
        elif self.rdepth > self.parent.rdepth:
            self.process_split()
        else:
            force_split = self.process_refine()

        # Set the estimated error
        if np.isinf(self.est_err):
            self.est_err = self.err
        ival = self.parent
        while ival is not None:
            # update the error estimates on all ancestor intervals
            children_err = sum(i.est_err for i in ival.children)
            if np.isfinite(children_err):
                ival.est_err = children_err
                ival = ival.parent
            else:
                break

        # Check whether the point spacing is smaller than machine precision
        # and pop the interval with the largest error and do not split
        remove = self.err < (abs(self.igral) * eps * Vcond[self.depth])
        if remove:
            # If this interval is discarded from ivals, there is no need
            # to split it further.
            force_split = False

        return force_split, remove

    def process_make_first(self):
        fx = np.array(self.done_points.values())
        nans = _zero_nans(fx)

        self.c[3] = V_inv[3] @ fx
        self.c[2, :ns[2]] = V_inv[2] @ fx[:ns[3]:2]
        fx[nans] = np.nan
        self.fx = fx

        self.c_old = np.zeros(fx.shape)
        c_diff = norm(self.c[self.depth] - self.c[2])

        a, b = self.a, self.b
        self.err = (b - a) * c_diff
        self.igral = (b - a) * self.c[self.depth, 0] / sqrt(2)

        if c_diff / norm(self.c[3]) > 0.1:
            self.err = max(self.err, (b-a) * norm(self.c[3]))

    def process_split(self, ndiv_max=20):
        fx = np.array(self.done_points.values())
        self.c[self.depth, :ns[self.depth]] = c_new = _calc_coeffs(fx, self.depth)
        self.fx = fx

        parent = self.parent
        self.c_old = self.T @ parent.c[parent.depth]
        c_diff = norm(self.c[self.depth] - self.c_old)

        a, b = self.a, self.b
        self.err = (b - a) * c_diff
        self.igral = (b - a) * self.c[self.depth, 0] / sqrt(2)

        self.ndiv = (parent.ndiv
                     + (abs(parent.c[0, 0]) > 0
                        and self.c[0, 0] / parent.c[0, 0] > 2))

        if self.ndiv > ndiv_max and 2*self.ndiv > self.rdepth:
            raise DivergentIntegralError(self)

    def process_refine(self):
        fx = np.array(self.done_points.values())
        self.c[self.depth, :ns[self.depth]] = c_new = _calc_coeffs(fx, self.depth)
        self.fx = fx

        c_diff = norm(self.c[self.depth - 1] - self.c[self.depth])

        a, b = self.a, self.b
        self.err = (b - a) * c_diff
        self.igral = (b - a) * c_new[0] / sqrt(2)
        nc = norm(c_new)
        force_split = nc > 0 and c_diff / nc > 0.1
        return force_split

    def __repr__(self):
        lst = [
            '(a, b)=({:.5f}, {:.5f})'.format(self.a, self.b),
            'depth={}'.format(self.depth),
            'rdepth={}'.format(self.rdepth),
            'err={:.5E}'.format(self.err),
            'igral={:.5E}'.format(self.igral if self.igral else 0),
            'est_err={:.5E}'.format(self.est_err),
            'discard={}'.format(self.discard),
        ]
        return ' '.join(lst)

    def equal(self, other, *, verbose=False):
        """Note: Implementing __eq__ breaks SortedContainers in some way."""
        if not self.complete:
            if verbose:
                print('Interval {} is not complete.'.format(self))
            return False

        slots = set(self.__slots__).intersection(other.__slots__)
        same_slots = []
        for s in slots:
            a = getattr(self, s)
            b = getattr(other, s)
            is_equal = np.allclose(a, b, rtol=0, atol=eps, equal_nan=True)
            if verbose and not is_equal:
                print('self.{} - other.{} = {}'.format(s, s, a - b))
            same_slots.append(is_equal)

        return all(same_slots)


class IntegratorLearner(BaseLearner):

    def __init__(self, function, bounds, tol=None, rtol=None):
        """
        Parameters
        ----------
        function : callable: X → Y
            The function to learn.
        bounds : pair of reals
            The bounds of the interval on which to learn 'function'.
        tol : float
            Absolute tolerance of the error to the integral, this means that the
            learner is done when: `tol > err`.
        rtol : float, optional
            Relative tolerance of the error to the integral, this means that the
            learner is done when: `tol > err / abs(igral)`. If `rtol` is
            provided, `tol` is optional. If both `rtol` and `tol`
            are provided, both conditions must be satisfied.

        Attributes
        ----------
        complete_branches : list of intervals
            The intervals that can be used in the determination of the integral.
        nr_points : int
            The total number of evaluated points.
        igral : float
            The integral value in `self.bounds`.
        err : float
            The absolute error associated with `self.igral`.

        Methods
        ------- 
        done : bool
            Returns whether the `tol` has been reached.
        plot : hv.Scatter
            Plots all the points that are evaluated.
        """
        self.function = function
        self.bounds = bounds
        self.tol = tol
        self.rtol = rtol
        if self.rtol is None and self.tol is None:
            raise ValueError('One of `tol` or `rtol` must be provided.')
        self.priority_split = []
        self.ivals = SortedSet([], key=attrgetter('err'))
        self.done_points = {}
        self.not_done_points = set()
        self._stack = []
        self._err_final = 0
        self._igral_final = 0
        self.x_mapping = defaultdict(lambda: SortedSet([], key=attrgetter('rdepth')))
        ival, points = Interval.make_first(*self.bounds, self.tol)
        self._update_ival(ival, points)
        self.first_ival = ival
        self._complete_branches = []

    def add_point(self, point, value):
        if point not in self.x_mapping:
            raise ValueError("Point {} doesn't belong to any interval"
                             .format(point))
        self.done_points[point] = value
        self.not_done_points.discard(point)

        # Select the intervals that have this point
        ivals = self.x_mapping[point]
        for ival in ivals:
            ival.done_points[point] = value
            if ival.complete and not ival.done and not ival.discard:
                in_ivals = ival in self.ivals
                self.ivals.discard(ival)
                force_split, remove = ival.complete_process()
                if remove:
                    self._err_final += ival.err
                    self._igral_final += ival.igral
                elif in_ivals:
                    self.ivals.add(ival)

                if force_split:
                    # Make sure that at the next execution of _fill_stack(),
                    # this ival will be split.
                    self.priority_split.append(ival)

    def _update_ival(self, ival, points):
        assert not ival.discard
        for x in points:
            # Update the mappings
            self.x_mapping[x].add(ival)
            if x in self.done_points:
                self.add_point(x, self.done_points[x])
            elif x not in self.not_done_points:
                self.not_done_points.add(x)
                self._stack.append(x)

        # Add the new interval to the err sorted set
        self.ivals.add(ival)

    def set_discard(self, ival):
        def _discard(ival):
            ival.discard = True
            self.ivals.discard(ival)
            for point in self._stack:
                # XXX: is this check worth it?
                if all(i.discard for i in self.x_mapping[point]):
                    self._stack.remove(point)
            for child in ival.children:
                _discard(child)
        _discard(ival)

    def choose_points(self, n):
        points, loss_improvements = self.pop_from_stack(n)
        n_left = n - len(points)
        while n_left > 0:
            assert n_left >= 0
            self._fill_stack()
            new_points, new_loss_improvements = self.pop_from_stack(n_left)
            points += new_points
            loss_improvements += new_loss_improvements
            n_left -= len(new_points)

        return points, loss_improvements

    def pop_from_stack(self, n):
        points = self._stack[:n]
        self._stack = self._stack[n:]
        loss_improvements = [max(ival.err for ival in self.x_mapping[x])
                             for x in points]
        return points, loss_improvements

    def remove_unfinished(self):
        pass

    def _fill_stack(self):
        # XXX: to-do if all the ivals have err=inf, take the interval
        # with the lowest rdepth and no children.
        if self.priority_split:
            ival = self.priority_split.pop()
            force_split = True
            if ival.children:
                # If the interval already has children (which is the result of an
                # earlier refinement when the data of the interval wasn't known
                # yet,) then discard the children and propagate it down.
                for child in ival.children:
                    self.set_discard(child)
        else:
            ival = self.ivals[-1]
            force_split = False
            assert not ival.children

        # Remove the interval from the err sorted set because we're going to
        # split or refine this interval
        self.ivals.discard(ival)

        # If the interval points are smaller than machine precision, then
        # don't continue with splitting or refining.
        points = ival.points(ival.depth)
        reached_machine_tol = points[1] <= points[0] or points[-1] <= points[-2]

        if (not ival.discard) and (not reached_machine_tol):
            if ival.depth == 3 or force_split:
                # Always split when depth is maximal or if refining didn't help
                ivals_new = ival.split()
                for ival_new in ivals_new:
                    points = ival_new.points(depth=0)
                    self._update_ival(ival_new, points)
            else:
                # Refine
                ival_new, points = ival.refine()
                self._update_ival(ival_new, points)

        # Remove the smallest element if number of intervals is larger than 1000
        if len(self.ivals) > 1000:
            self.ivals.pop(0)

        return self._stack

    @staticmethod
    def deepest_complete_branches(ival):
        """Finds the deepest complete set of intervals starting from `ival`."""
        complete_branches = []
        def _find_deepest(ival):
            children_err = (sum(i.est_err for i in ival.children)
                            if ival.children else np.inf)
            if np.isfinite(ival.est_err) and np.isinf(children_err):
                complete_branches.append(ival)
            else:
                for i in ival.children:
                    _find_deepest(i)
        _find_deepest(ival)
        return complete_branches

    @property
    def complete_branches(self):
        if not self.first_ival.done:
            return []

        if not self._complete_branches:
            self._complete_branches.append(self.first_ival)

        complete_branches = []
        for ival in self._complete_branches:
            if ival.discard:
                complete_branches = self.deepest_complete_branches(self.first_ival)
                break
            if not ival.children:
                # If the interval has no children, than is already is the deepest
                # complete branch.
                complete_branches.append(ival)
            else:
                complete_branches.extend(self.deepest_complete_branches(ival))
        self._complete_branches = complete_branches
        return self._complete_branches

    @property
    def nr_points(self):
        return len(self.done_points)

    @property
    def igral(self):
        return sum(i.igral for i in self.complete_branches)

    @property
    def err(self):
        complete_branches = self.complete_branches
        if not complete_branches:
            return np.inf
        else:
            return sum(i.err for i in complete_branches)

    def done(self):
        err = self.err
        igral = self.igral
        if self.tol is not None:
            is_done = (err == 0
                    or err < self.tol
                    or (self._err_final > self.tol
                        and err - self._err_final < self.tol)
                    or not self.ivals)
        else:
            is_done = True
        if self.rtol is not None:
            is_rdone = (err == 0
                    or err < abs(igral) * self.rtol
                    or (self._err_final > abs(igral) * self.rtol
                        and err - self._err_final < abs(igral) * self.rtol)
                    or not self.ivals)
        else:
            is_rdone = True
        return is_done and is_rdone

    def loss(self, real=True):
        return abs(abs(self.igral) * self.tol - self.err)

    def equal(self, other, *, verbose=False):
        """Note: `other` is a list of ivals."""
        if len(self.ivals) != len(other):
            if verbose:
                print('len(self.ivals)={} != len(other)={}'.format(
                    len(self.ivals), len(other)))
            return False

        ivals = [sorted(i, key=attrgetter('a')) for i in [self.ivals, other]]
        return all(ival.equal(other_ival, verbose=verbose)
                   for ival, other_ival in zip(*ivals))

    def plot(self):
        return hv.Scatter(self.done_points)
