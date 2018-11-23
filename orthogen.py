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
from numpy import matmul, multiply, ones, zeros, eye
from numpy.linalg import inv, norm

class OrthoDataGen(object):
    '''
    Orthogonal Data Generator

    Rawdata consists of data matrix X and Z where Z might be highly correlated
    with X due to some potential bias. The objective of this method is removing
    the bias by looking for X_hat is orthogonal to Z (cov(X_hat, Z) = 0) with
    minimal information loss w.r.t X.

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
        self.n, self.p, self.k = n, p, k # number of data / number of feature / rank
        self.X = X # n x p data matrix of p features measured over n subjects,
        self.Z = Z # n x m additional group membership variable,
        self.U = np.random.uniform(0, 1, (p, k)) # p x k matrix of k linear orthonormal basis vectors,
        self.S = np.random.uniform(0, 1, (n, k)) # n x k matrix of associated scores.
        print('[%s] n = %d, p = %d, k = %d, X is %d x %d, Z is %d x %d.' %
            (arrow.now(), n, p, k, n, p, n, m), file=sys.stderr)

    def sog(self, t, tol=1e-1, verbose=True):
        '''
        Sparse Orthogonal to Subgroup (SOG) algorithm
        '''
        THETA_DETAULT = 1.
        # update each column of matrix U and S.
        for j in range(self.k):
            print('[%s] updating %d/%d ...' % (arrow.now(), j+1, self.k), file=sys.stderr)
            # stop when changes in u_j and s_j are sufficiently small
            last_s_j = zeros((self.n, 1))
            last_u_j = zeros((self.p, 1))
            i = 0
            while self.change_measure(last_s_j, self.S[:, j], tol) or \
                  self.change_measure(last_u_j, self.U[:, j], tol):
                # update last s_j, u_j
                last_s_j     = np.copy(self.S[:, j])
                last_u_j     = np.copy(self.U[:, j])
                P_j          = eye(self.n) - sum([ matmul(self.S[:, l], self.S[:, l].T) for l in range(j-1) ])
                beta_j       = matmul(matmul(matmul(matmul(inv(matmul(self.Z.T, self.Z)), self.Z.T), P_j), self.X), self.U[:, j])
                # update s_j
                # s_j is the column of matrix U, where len(s_j) = n, j = 1, ..., k
                unnorm_s_j   = matmul(matmul(P_j, self.X), self.U[:, j]) - matmul(self.Z, beta_j)
                self.S[:, j] = unnorm_s_j / norm(unnorm_s_j + 1e-10)
                # update u_j
                # u_j is the column of matrix S, where len(u_j) = p, j = 1, ..., k
                XT_sj        = matmul(self.X.T, self.S[:, j])
                theta        = 0 if norm(XT_sj, ord=1) <= t else THETA_DETAULT
                S_theta_x    = self.soft_threshold_operator(theta, XT_sj)
                self.U[:, j] = S_theta_x / norm(S_theta_x + 1e-10)
                # log information if verbose is true
                if verbose:
                    print('[%s] ---------------------------------' % arrow.now(), file=sys.stderr)
                    print('[%s] iter %d' % (arrow.now(), i), file=sys.stderr)
                    print('[%s]\t||u_j||_1 = %.3f' %
                        (arrow.now(), norm(self.U[:, j], ord=1)),
                        file=sys.stderr)
                    print('[%s]\ts_j change is %.3f, u_j change is %.3f' %
                        (arrow.now(),
                         norm(last_s_j - self.S[:, j]),
                         norm(last_u_j - self.U[:, j])),
                        file=sys.stderr)
                    print('[%s]\tFrobenius measure is %.3f' %
                        (arrow.now(), norm(self.X - matmul(self.S, self.U.T), ord='fro')),
                        file=sys.stderr)
                    i += 1

    def reconstruct(self):
        '''
        Reconstruct X_hat
        '''
        # d_j = s_j^T * X * u_j.
        d = [ (matmul(matmul(self.S[:, j].T, self.X), self.U[:, j]) * self.S[:, j]).tolist()
            for j in range(self.k) ]
        # S denote the n x k matrix with columns d_j * s_j.
        # U is the p x k sparse matrix with rows u_j, j = 1, ..., k
        # return X_hat = S * U^T
        X_hat = matmul(np.array(d).T, self.U.T)
        return X_hat

    @staticmethod
    def soft_threshold_operator(theta, x):
        indicator_func = ones(len(x)) * (abs(x) >= theta)
        return multiply(multiply(np.sign(x), (abs(x) - theta)), indicator_func)

    @staticmethod
    def change_measure(mat_a, mat_b, tol):
        measure = norm(mat_a - mat_b, ord=2)
        return True if measure > tol else False



# Unittest on a simple example
if __name__ == '__main__':
    X = np.array([
        [1, 1, 0, 0.1, 0,   0],
        [1, 1, 0, 0,   0,   0],
        [0, 0, 1, 1,   0,   0],
        [0, 0, 1, 1,   0.1, 0],
        [0, 0, 0, 0,   1,   1]
    ])
    Z = np.array([
        [1,   1],
        [1,   1],
        [1,   1.1],
        [1.1, 1],
        [1,   1.1]
    ])
    k   = 3
    odg = OrthoDataGen(X, Z, k)
    odg.sog(t=1.529, tol=1e-2)
    print(odg.reconstruct())
