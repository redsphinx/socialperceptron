import numpy as np
import os
import deepimpression2.chalearn30.data_utils as D
import deepimpression2.paths as P


def convert(pred_path, trait):
    all_traits = ['O', 'C', 'E', 'A', 'S']
    assert(trait in all_traits)

    pred_path = os.path.join(P.LOG_BASE, pred_path)

    idx = all_traits.index(trait)

    if not os.path.exists(pred_path):
        print('wrong prediction folder')
        return

    pred = np.genfromtxt(pred_path, dtype=float)
    labels = D.basic_load_personality_labels('test')[:, idx]

    loss = np.abs(pred - labels)

    print('mean loss trait %s: %f' % (trait, np.mean(loss)))



tr_all = ['O', 'C', 'E', 'A', 'S']

for tr in tr_all:
    convert('pred_87_%s.txt' % tr, tr)
