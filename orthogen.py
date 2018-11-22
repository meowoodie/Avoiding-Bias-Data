#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Orthogonal data generator for data transformation. After transformation, the
specified group of features will be orthogonal to rest of features with minimal
information loss.
'''

import sys
import arrow
import numpy as np
from numpy import dot, ones, zeros
from numpy.linalg import inv, norm

class OrthoDataGen(object):
    '''
    Orthogonal Data Generator

    Rawdata consists of data matrix X and Z where Z might be highly correlated
    with X due to some potential bias. The objective of this method is removing
    the bias by looking for X_hat is orthogonal to Z (cov(X_hat, Z) = 0) with minimal
    information loss w.r.t X.

    This problem is equivalent to a minimization of Frobenius distance.
    argmin || X - SU^T ||_F^2, subject to < SU^T, Z > = 0, U \\in G_{p,k}
    where G_{p,k} is the Grassman maniford of orthogonal matrices.
    '''

    def __init__(self, X, Z, k):
        # data dimension check
        n, p   = X.shape
        n_z, m = Z.shape
        assert n == n_z, 'X and Z have unequal number of rows.'
        # variables initialization
        self.p, self.n = p, n
        self.k = k
        self.X = X # n x p data matrix of p features measured over n subjects,
        self.Z = Z # n x ? additional group membership variable,
        self.U = np.random.uniform(0, 1, (p, k)) # p x k matrix of k linear orthonormal basis vectors,
        self.S = np.random.uniform(0, 1, (n, k)) # n x k matrix of associated scores.

    def sog(self, t, tol=1e-2):
        '''
        Sparse Orthogonal to Subgroup (SOG) algorithm
        '''
        THETA_DETAULT = 1.
        # update each column of matrix U and S.
        for j in range(self.k):
            print('[%s] updating %d/%d ...', % (arrow.now(), j, self.k), file=sys.stderr)
            # stop when changes in u_j and s_j are sufficiently small
            last_s_j = zeros((self.n, 1))
            last_u_j = zeros((self.p, 1))
            while self.frobenius_measure(last_s_j, self.S[:, j], tol) and \
                  self.frobenius_measure(last_u_j, self.U[:, j], tol):
                # update last s_j, u_j
                last_s_j     = self.S[:, j]
                last_u_j     = self.U[:, j]
                P_j          = ones((n, n)) - sum([ dot(self.S[:, l], self.S[:, l].T) for l in range(j-1) ])
                beta_j       = dot(dot(dot(dot(inv(dot(self.Z.T, self.Z)), self.Z.T), P_j), self.X), self.U[:, j])
                # update s_j
                # s_j is the column of matrix U, which len(s_j) = n, j = 1, ..., k
                unnorm_s_j   = (dot(dot(P_j, self.X), self.U[:, j]) - beta_j * self.Z)
                self.S[:, j] = unnorm_s_j / norm(unnorm_s_j)
                # update u_j
                # u_j is the column of matrix S, which len(u_j) = p, j = 1, ..., k
                XT_sj        = dot(self.X.T, self.S[:, j])
                theta        = 0 if norm(XT_sj, ord=1) < t else THETA_DETAULT
                S_theta_x    = self.soft_threshold_operator(theta, XT_sj)
                self.U[:, j] = S_theta_x / norm(S_theta_x)
                print('[%s] ||u_j||_1 = %.3f', % (arrow.now(), norm(self.U[:, j], ord=1)), file=sys.stderr)

    @staticmethod
    def soft_threshold_operator(theta, x):
        indicator_func = ones(len(x)) * (abs(x) >= theta)
        return np.sign(x) * (abs(x) - theta) * indicator_func

    @staticmethod
    def frobenius_measure(mat_a, mat_b, tol, verbose=True):
        measure = norm(mat_a - mat_b, ord='fro')
        if verbose:
            print('[%s] frobenius measurement is %.3f', % (arrow.now(), measure), file=sys.stderr)
        return True if measure > tol else False



if __name__ == '__main__':
    X = np.array([

    ])
    Z = np.array([

    ])
    odg = OrthoDataGen(X, Z, k)
    odg.sog(t=3, tol=1e-2)
