import numpy as np
from numba import njit
from functools import cache

@cache
def A(n, c):
    if n <= c:
        return 2**n
    else:
        s = 0
        for j in range(c, -1, -1):
            s += A(n - 1 - j, c)
        return s


def proba_longest_run(n, c):
    """
    Probability to obtain a run of more than c consecutive head or tails
    """
    return 1.0 - A(n-1, c-1) / 2**(n-1)

@njit
def longest_run(x):
    """
    Find longest sequence of positive or negative numbers
    """
    d = 0
    last = 0
    length = 0
    longest = 0
    for i in range(len(x)):
        if x[i] > 0:
            d = 1
        elif x[i] < 0:
            d = -1
        else:
            d = 0
            
        if d * last <= 0:
            longest = max(longest, abs(length))
            length = d
        else:
            length += d
        last = d
        
    longest = max(longest, abs(length))
    return longest

def pval(a, b):
    """
    Calculate the probability for a and b to be equivalent 
    """
    diff = b - a
    n = diff.size
    # adjust c because we calculate P(Rn > C) 
    # but be want probability for >= c so calculate for c-1
    c = longest_run(diff) - 1
    return proba_longest_run(n, c)

def make_cormap(data):
    n = len(data)
    cormap = np.zeros((n, n))
    # Bonferroni Correction
    correction = sum(range(n))
    for i in range(n):
        for j in range(i, n):
            # adjusted p value
            p = pval(data[i], data[j]) * correction
            p = min(1.0, p)
            cormap[i, j] = p
            cormap[j, i] = p
    return cormap

assert(longest_run(np.array([1, 2, 3, -5, -5, 4, 3, -5, -1, -2, -4])) == 4)
assert(longest_run(np.array([-1, -1, -1, 0, 0, 0])) == 3)
assert(longest_run(np.array([1, 1, 1, 0, 0, 0])) == 3)
assert(longest_run(np.array([0, 0, -1, 0, 0, 0])) == 1)

assert(A(6, 3) == 56)
assert(A(7, 3) == 108)
assert(A(8, 3) == 208)
assert(proba_longest_run(9, 3) == 0.41796875)
assert(proba_longest_run(9, 4) == 0.1875)