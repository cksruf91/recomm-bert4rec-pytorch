import sys
import time
from datetime import datetime

import numpy as np
import scipy.sparse as sp


def to_sparse_matrix(df, num_x, num_y, x_col, y_col, v_col):
    total = len(df)

    mat = sp.dok_matrix((num_x + 1, num_y + 1), dtype=np.float32)
    for i, (user, item, rating) in enumerate(zip(df[x_col], df[y_col], df[v_col])):
        progressbar(total, i + 1, suffix='to sparse matrix')
        if rating > 0:
            mat[user, item] = 1.0

    return mat


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class DefaultDict(dict):
    """KeyError -> return default value"""

    def __init__(self, default, *arg):
        super().__init__(*arg)
        self.default = default

    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            return self.default


def progressbar(total, i, bar_length=50, prefix='', suffix=''):
    """progressbar
    """
    bar_graph = '█'
    if i % max((total // 100), 1) == 0:
        dot_num = int((i + 1) / total * bar_length)
        dot = bar_graph * dot_num
        empty = '.' * (bar_length - dot_num)
        sys.stdout.write(f'\r {prefix} [{dot}{empty}] {i / total * 100:3.2f}% {suffix}')
    if i == total:
        sys.stdout.write(f'\r {prefix} [{bar_graph * bar_length}] {100:3.2f}% {suffix}')
        print(' Done')


def to_timestampe(x, format_string):
    return int(time.mktime(datetime.strptime(x, format_string).timetuple()))
