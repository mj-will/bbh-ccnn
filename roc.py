#!/bin/usr/env python

import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from sklearn import metrics

# load results
# 0 ->  2 CCNN layers
preds0  = np.array(np.load('./results0.npy'))
signal_preds0 = np.asarray([y[1] for x in preds0 for y in x])
# 1 -> 1 CCNN layer
preds1  = np.array(np.load('./results1.npy'))
signal_preds1 = np.asarray([y[1] for x in preds1 for y in x])
# 2 -> CNN
preds2  = np.array(np.load('./results5.npy'))
signal_preds2 = np.asarray([y[1] for x in preds2 for y in x])
# 2 -> 1 CCNN layer
preds3  = np.array(np.load('./results4.npy'))
signal_preds3 = np.asarray([y[1] for x in preds3 for y in x])


# load targets
targets = np.load('./targets.npy')

fpr0, tpr0, _= metrics.roc_curve(targets, signal_preds0)
fpr1, tpr1, _= metrics.roc_curve(targets, signal_preds1)
fpr2, tpr2, _= metrics.roc_curve(targets, signal_preds2)
fpr3, tpr3, _= metrics.roc_curve(targets, signal_preds3)

fig = plt.figure()
plt.plot(fpr0, tpr0, label = 'CCNN - 2 layers')
plt.plot(fpr1, tpr1, label = 'CCNN - 1 layer')
plt.plot(fpr3, tpr3, label = 'CCNN - 1 layer')
plt.plot(fpr2, tpr2, label = 'CNN')
plt.title('SNR 8 - 1K')
plt.xlabel('FDR')
plt.grid()
plt.xlim(0.0, 0.4)
plt.ylim(0.6,1.0)
plt.ylabel('TDR')
plt.legend(loc = 'lower right')
fig.savefig('roc.png')

