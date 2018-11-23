#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Test on real data set: Text documents in Atlanta Police 911 Calls Crime Reports.
'''

import arrow
import utils
import orthogen
import numpy as np

if __name__ == '__main__':
    # keywords that we are studying
    X_KEYWORDS   = ['stole', 'robbery', 'males']
    Z_KEYWORDS   = ['black', 'black_males']

    # load raw data matrix X (exclude biased keywords) and subgroup variable Z (biased keywords)
    _, XZ = utils.extract_keywords_from_corpus(keywords=all_keyword)

    # configurations
    all_keyword   = X_KEYWORDS + Z_KEYWORDS
    x_col_idx     = [ ALL_KEYWORDS.index(keyword) for keyword in X_KEYWORDS ]
    z_col_idx     = [ ALL_KEYWORDS.index(keyword) for keyword in Z_KEYWORDS ]
    valid_row_idx = np.where(XZ.sum(axis=1) > 0)[0]

    # remove rows with all zero entries
    XZ = XZ[valid_row_idx, :]
    print('[%s] raw data matrix is %d x %d' % (arrow.now(), XZ.shape[0], XZ.shape[1]), file=sys.stderr)
    # split XZ into X and Z according to their keywords
    X = XZ[:, x_col_idx]
    Z = XZ[:, z_col_idx]
    # initiate orthogonal data generator
    odg = orthogen.OrthoDataGen(X, Z, k=2)
    odg.sog(t=1.414, tol=1e-2)
    # X_hat = odg.reconstruct()
