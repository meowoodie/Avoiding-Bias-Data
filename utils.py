#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Utilities for preprocessing test data
'''

import sys
import arrow
import gensim
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages
from collections import defaultdict
from gensim import corpora, models

def extract_keywords_from_corpus(
    keywords,
    dict_path='/Users/woodie/Desktop/workspace/Event-Series-Detection/resource/dict/10k.bigram.dict',
    corpus_path='/Users/woodie/Desktop/workspace/Event-Series-Detection/resource/corpus/10k.bigram.tfidf.corpus',
    result_path='data/10.biased.keywords.txt',
    savetxt=True):
    # get dictionary
    ngram_dict = corpora.Dictionary.load(dict_path)
    print(ngram_dict, file=sys.stderr)
    # get corpus
    corpus_tfidf = corpora.MmCorpus(corpus_path)
    print(corpus_tfidf, file=sys.stderr)

    # convert to dense corpus if necessary
    dense_corpus = gensim.matutils.corpus2dense(corpus_tfidf, num_terms=len(ngram_dict)).transpose()
    keyword_ids = [ ngram_dict.token2id[keyword] for keyword in keywords]
    print('extraced keywords ids: %s' % keyword_ids, file=sys.stderr)
    keywords_mat    = dense_corpus[:, keyword_ids]
    nonkeywords_mat = np.delete(dense_corpus, keyword_ids, axis=1)
    if savetxt:
        np.savetxt(result_path, keywords_mat, delimiter=',')
    return nonkeywords_mat, keywords_mat


if __name__ == '__main__':

    extract_keywords_from_corpus(keywords=[
        'burglary', 'robbery', 'carjacking', 'stole', 'jewelry', 'arrestee', 'jail',
        'shot', 'black', 'male', 'males', 'black_male', 'black_males'])
