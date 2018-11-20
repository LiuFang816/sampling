# encoding: utf-8
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn
index2word = ["你","好","合","协"]

def sample_discrete(vec):
    u = np.random.rand()
    start = 0
    for i, num in enumerate(vec):
        if u > start:
            start += num
        else:
            return i-1
    return i

count = dict([(w, 0) for w in index2word])
# 采样1000次
for i in range(1000):
    s = sample_discrete([0.1, 0.5, 0.2, 0.2])
    count[index2word[s]] += 1
for k in count:
    print(k," : ", count[k])

