# The Selector library provides a set of tools for selecting a
# subset of the dataset and computing diversity.
#
# Copyright (C) 2023 The QC-Devs Community
#
# This file is part of Selector.
#
# Selector is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# Selector is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --
"""Module for Distance-Based Selection Methods."""
import numpy as np
from selector.methods.base import SelectionBase
from selector.methods.utils import optimize_radius
from sklearn.metrics.pairwise import euclidean_distances

__all__ = [
    "MaxMin",
    "MaxSum",
    "OptiSim",
    "DISE",
]


class MaxMin(SelectionBase):
    """Select samples using MaxMin algorithm.

    MaxMin is possibly the most widely used method for dissimilarity-based
    compound selection. When presented with a dataset of samples, the
    initial point is chosen as the dataset's medoid center. Next, the second
    point is chosen to be that which is furthest from this initial point.
    Subsequently, all following points are selected via the following
    logic:

    1. Find the minimum distance from every point to the already-selected ones.
    2. Select the point which has the maximum distance among those calculated
       in the previous step.

    In the current implementation, this method requires or computes the full pairwise-distance
    matrix, so it is not recommended for large datasets.

    References
    ----------
    [1] Ashton, Mark, et al., Identification of diverse database subsets using
    property‐based and fragment‐based molecular descriptions, Quantitative
    Structure‐Activity Relationships 21.6 (2002): 598-604.
    """

    def __init__(self, fun_dist = euclidean_distances):
        """
        Initializing class.

        Parameters
        ----------
        fun_distance : callable
            Function for calculating the pairwise distance between sample points.
            `fun_dist(X) -> X_dist` takes a 2D feature array of shape (n_samples, n_features)
            and returns a 2D distance array of shape (n_samples, n_samples).
        """
        self.fun_dist = fun_dist

    def select_from_cluster(self, X, size, seed):
        """Return selected samples from a cluster based on MaxMin algorithm.

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features) or (n_samples, n_samples)
            Feature matrix of `n_samples` samples in `n_features` dimensional feature space,
            or the pairwise distance matrix between `n_samples` samples.
            If `fun_dist` is `None`, the `X` is assumed to be a square pairwise distance matrix.
        size: int
            Number of sample points to select (i.e. size of the subset).
        seed: int
            index used to initialize MDA

        Returns
        -------
        selected : list
            List of indices of selected samples.
        """
        # calculate pairwise distance between points
        X_dist = X
        if self.fun_dist is not None:
            X_dist = self.fun_dist(X)
        # check X_dist is a square symmetric matrix
        if X_dist.shape[0] != X_dist.shape[1]:
            raise ValueError(f"The pairwise distance matrix must be square, got {X_dist.shape}.")
        if np.max(abs(X_dist - X_dist.T)) > 1e-8:
            raise ValueError("The pairwise distance matrix must be symmetric.")
            
        # initialize MDA with an initial point
        selected = [seed]
        # select following points until desired number of points have been obtained
        while len(selected) < size:
            # determine the min pairwise distances between the selected points and all other points
            min_distances = np.min(X_dist[selected], axis=0)
            # determine which point affords the maximum distance among the minimum distances
            # captured in min_distances
            new_id = np.argmax(min_distances)
            selected.append(new_id)
        return selected
