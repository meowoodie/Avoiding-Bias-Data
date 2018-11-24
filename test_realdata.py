#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Test on real data set: Text documents in Atlanta Police 911 Calls Crime Reports.
'''

import sys
import arrow
import utils
import orthogen
import numpy as np

if __name__ == '__main__':
    # keywords that we are studying
    X_KEYWORDS   = ['burglary', 'robbery', 'carjacking', 'stole', 'jewelry', 'arrestee', 'shot']
    Z_KEYWORDS   = ['black_male', 'black_males']
    C_KEYWORDS   = ['black', 'male', 'males']    # for comparison, no changes will be made on this group
    ALL_KEYWORDS = X_KEYWORDS + C_KEYWORDS + Z_KEYWORDS

    # load raw data matrix X (exclude biased keywords) and subgroup variable Z (biased keywords)
    _, XZ = utils.extract_keywords_from_corpus(keywords=ALL_KEYWORDS, savetxt=False)

    # configurations
    x_col_idx     = [ ALL_KEYWORDS.index(keyword) for keyword in X_KEYWORDS ]
    z_col_idx     = [ ALL_KEYWORDS.index(keyword) for keyword in Z_KEYWORDS ]
    c_col_idx     = [ ALL_KEYWORDS.index(keyword) for keyword in C_KEYWORDS ]
    valid_row_idx = np.where(XZ.sum(axis=1) > 0)[0]

    # remove rows with all zero entries
    XZ = XZ[valid_row_idx, :]
    print('[%s] raw data matrix is %d x %d' % (arrow.now(), XZ.shape[0], XZ.shape[1]), file=sys.stderr)
    # split XZ into X and Z according to their keywords
    X = XZ[:, x_col_idx]
    Z = XZ[:, z_col_idx]
    C = XZ[:, c_col_idx]
    # initiate orthogonal data generator
    odg = orthogen.OrthoDataGen(X, Z, k=2)
    odg.sog(t=1.5, tol=1e-1)
    print(odg.S)
    print(odg.U)
    X_hat = odg.reconstruct()

    # save results
    result = np.concatenate([X_hat, C, Z], axis=1)
    np.savetxt('data/debug.recon.9.biased.keywords.txt', result, delimiter=',')
