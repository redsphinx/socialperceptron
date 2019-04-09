import numpy as np
import os
from scipy import stats
import h5py as h5
from shutil import copyfile
from tqdm import tqdm

import deepimpression2.paths as P
import deepimpression2.chalearn30.data_utils as D


def find_index():
    key_where = '4ZlcaXadwlo.005.mp4'
    labels = h5.File(P.CHALEARN_TEST_LABELS_20, 'r')
    for i, k in enumerate(labels.keys()):
        if k == key_where:
            print(i)
            break

# find_index()


def fix_labels(labels):
    ignore = 173
    labels = np.delete(labels, ignore, axis=0)
    return labels


def correlations_ground_truth(trait, name):
    traits_all = ['O', 'C', 'E', 'A', 'S']
    index = traits_all.index(trait)
    target = D.basic_load_personality_labels('test')
    target = target[:, index]
    # target = fix_labels(target)

    path = os.path.join(P.LOG_BASE, name)
    predictions = np.genfromtxt(path, delimiter=',', dtype=float)

    r, p = stats.pearsonr(predictions, target)

    print(name, 'r: ', r, 'p: ', p)


def corr_face_single_deepimpression():
    traits = ['O', 'C', 'E', 'A', 'S']
    # models = ['pred_109_O.txt', 'pred_109_C.txt', 'pred_109_E.txt', 'pred_109_A.txt', 'pred_109_S.txt']  # frame=6
    models = ['pred_132_O.txt', 'pred_132_C.txt', 'pred_132_E.txt', 'pred_132_A.txt', 'pred_132_S.txt']  # frame=30

    for i in range(5):
        correlations_ground_truth(traits[i], models[i])


# corr_face_single_deepimpression()


def corr_bg_single_deepimpression():
    traits = ['O', 'C', 'E', 'A', 'S']
    # models = ['pred_110_O.txt', 'pred_110_C.txt', 'pred_110_E.txt', 'pred_110_A.txt', 'pred_110_S.txt']  # frame=6
    models = ['pred_133_O.txt', 'pred_133_C.txt', 'pred_133_E.txt', 'pred_133_A.txt', 'pred_133_S.txt']  # frame=30

    for i in range(5):
        correlations_ground_truth(traits[i], models[i])

# corr_bg_single_deepimpression()

def corr_all_single_deepimpression():
    traits = ['O', 'C', 'E', 'A', 'S']
    # models = ['pred_112_O.txt', 'pred_112_C.txt', 'pred_112_E.txt', 'pred_112_A.txt', 'pred_112_S.txt']  # frame=6
    # models = ['pred_118_O.txt', 'pred_118_C.txt', 'pred_118_E.txt', 'pred_118_A.txt', 'pred_118_S.txt']  # frame=30
    # models = []  # frame=10
    models = ['pred_148_O.txt', 'pred_148_C.txt', 'pred_148_E.txt', 'pred_148_A.txt', 'pred_148_S.txt']  # frame=10

    correlations_ground_truth(traits[0], models[0])
    correlations_ground_truth(traits[1], models[1])
    correlations_ground_truth(traits[2], models[2])
    # correlations_ground_truth(traits[3], models[3])
    correlations_ground_truth(traits[4], models[4])


    # for i in range(5):
    #     correlations_ground_truth(traits[i], models[i])
# 
# corr_all_single_deepimpression()


def correlations_resnet_ground_truth(trait, name):
    traits_all = ['O', 'C', 'E', 'A', 'S']
    index = traits_all.index(trait)
    target = D.basic_load_personality_labels('test')
    target = target[:, index]
    target = fix_labels(target)

    path = os.path.join(P.LOG_BASE, name)
    predictions = np.genfromtxt(path, delimiter=',', dtype=float)
    predictions = predictions[:, index]

    r, p = stats.pearsonr(predictions, target)

    print(name, 'r: ', r, 'p: ', p)


def corr_face_resnet():
    traits = ['O', 'C', 'E', 'A', 'S']
    for i in range(5):
        correlations_resnet_ground_truth(traits[i], 'pred_114.txt')


def corr_bg_resnet(num):
    traits = ['O', 'C', 'E', 'A', 'S']
    for i in range(5):
        correlations_resnet_ground_truth(traits[i], 'pred_%d.txt' % num)


# corr_bg_resnet(171)



def how_many_frames():
    total = 0
    id_frames = h5.File(P.NUM_FRAMES, 'r')
    labels = h5.File(P.CHALEARN_TEST_LABELS_20, 'r')
    for k in labels.keys():
        k = k.split('.mp4')[0]
        total += id_frames[k][0]
    # 309973
    p = '/scratch/users/gabras/data/chalearn30/all_data'

    totsize = 0
    for k in labels.keys():
        name = k.split('.mp4')[0] + '.h5'
        path = os.path.join(p, name)
        totsize += os.path.getsize(path)

    # 252936585160/1024/1024/1024 = 236GB

    for k in tqdm(labels.keys()):
        name = k.split('.mp4')[0] + '.h5'
        d = '/home/gabras/chalearn_test_data'
        src = os.path.join(p, name)
        dst = os.path.join(d, name)
        copyfile(src, dst)


# how_many_frames()


def make_avg_preds():
    # -----------------------------------------------------------------------------
    # deepimpression
    face = ['pred_132_O', 'pred_132_C', 'pred_132_E', 'pred_132_A', 'pred_132_S']
    bg = ['pred_133_O', 'pred_133_C', 'pred_133_E', 'pred_133_A', 'pred_133_S']
    f_bg = ['pred_134_O', 'pred_134_C', 'pred_134_E', 'pred_134_A', 'pred_134_S']
    ff = 'pred_172'

    mean_face = 'pred_175'
    mean_bg = 'pred_176'
    mean_f_bg = 'pred_177'
    mean_ff = 'pred_178'
    
    # FACE
    mean_f_arr = np.zeros(1675)
    for i in face:
        p = os.path.join(P.LOG_BASE, i + '.txt')
        tmp = np.genfromtxt(p, delimiter=',', dtype=float)
        mean_f_arr += tmp
    
    mean_f_arr /= 5
    p = os.path.join(P.LOG_BASE, mean_face + '.txt')
    with open(p, 'a') as my_file:
        for i in mean_f_arr:
            line = '%f,\n' % i
            my_file.write(line)
    
    # BACKGROUND
    mean_bg_arr = np.zeros(1675)
    for i in bg:
        p = os.path.join(P.LOG_BASE, i + '.txt')
        tmp = np.genfromtxt(p, delimiter=',', dtype=float)
        mean_bg_arr += tmp

    mean_bg_arr /= 5
    p = os.path.join(P.LOG_BASE, mean_bg + '.txt')
    with open(p, 'a') as my_file:
        for i in mean_bg_arr:
            line = '%f,\n' % i
            my_file.write(line)
    
    # FACE + BACKGROUND
    mean_fbg_arr = np.zeros(1675)
    for i in f_bg:
        p = os.path.join(P.LOG_BASE, i + '.txt')
        tmp = np.genfromtxt(p, delimiter=',', dtype=float)
        mean_fbg_arr += tmp

    mean_fbg_arr /= 5
    p = os.path.join(P.LOG_BASE, mean_f_bg + '.txt')
    with open(p, 'a') as my_file:
        for i in mean_fbg_arr:
            line = '%f,\n' % i
            my_file.write(line)
    
    # FULL FRAME
    p = os.path.join(P.LOG_BASE, ff + '.txt')
    tmp = np.genfromtxt(p, delimiter=',', dtype=float)
    mean_ff_arr = np.mean(tmp, axis=1)

    p = os.path.join(P.LOG_BASE, mean_ff + '.txt')
    with open(p, 'a') as my_file:
        for i in mean_ff_arr:
            line = '%f,\n' % i
            my_file.write(line)
    # -----------------------------------------------------------------------------
    # resnet18, NP
    face = 'pred_169'
    bg = 'pred_170'
    f_bg = 'pred_171'
    ff = 'pred_174'

    mean_face = 'pred_183'
    mean_bg = 'pred_184'
    mean_f_bg = 'pred_185'
    mean_ff = 'pred_186'

    # FACE
    p = os.path.join(P.LOG_BASE, face + '.txt')
    tmp = np.genfromtxt(p, delimiter=',', dtype=float)
    mean_arr = np.mean(tmp, axis=1)

    p = os.path.join(P.LOG_BASE, mean_face + '.txt')
    with open(p, 'a') as my_file:
        for i in mean_arr:
            line = '%f,\n' % i
            my_file.write(line)

    # BG
    p = os.path.join(P.LOG_BASE, bg + '.txt')
    tmp = np.genfromtxt(p, delimiter=',', dtype=float)
    mean_arr = np.mean(tmp, axis=1)

    p = os.path.join(P.LOG_BASE, mean_bg + '.txt')
    with open(p, 'a') as my_file:
        for i in mean_arr:
            line = '%f,\n' % i
            my_file.write(line)

    # FACE + BG
    p = os.path.join(P.LOG_BASE, f_bg + '.txt')
    tmp = np.genfromtxt(p, delimiter=',', dtype=float)
    mean_arr = np.mean(tmp, axis=1)

    p = os.path.join(P.LOG_BASE, mean_f_bg + '.txt')
    with open(p, 'a') as my_file:
        for i in mean_arr:
            line = '%f,\n' % i
            my_file.write(line)

    # FULL FRAMES
    p = os.path.join(P.LOG_BASE, ff + '.txt')
    tmp = np.genfromtxt(p, delimiter=',', dtype=float)
    mean_arr = np.mean(tmp, axis=1)

    p = os.path.join(P.LOG_BASE, mean_ff + '.txt')
    with open(p, 'a') as my_file:
        for i in mean_arr:
            line = '%f,\n' % i
            my_file.write(line)
    # -----------------------------------------------------------------------------
    # rn18, pt
    face = 'pred_166'
    bg = 'pred_167'
    f_bg = 'pred_168'
    ff = 'pred_173'

    mean_face = 'pred_179'
    mean_bg = 'pred_180'
    mean_f_bg = 'pred_181'
    mean_ff = 'pred_182'

    # FACE
    p = os.path.join(P.LOG_BASE, face + '.txt')
    tmp = np.genfromtxt(p, delimiter=',', dtype=float)
    mean_arr = np.mean(tmp, axis=1)

    p = os.path.join(P.LOG_BASE, mean_face + '.txt')
    with open(p, 'a') as my_file:
        for i in mean_arr:
            line = '%f,\n' % i
            my_file.write(line)

    # BG
    p = os.path.join(P.LOG_BASE, bg + '.txt')
    tmp = np.genfromtxt(p, delimiter=',', dtype=float)
    mean_arr = np.mean(tmp, axis=1)

    p = os.path.join(P.LOG_BASE, mean_bg + '.txt')
    with open(p, 'a') as my_file:
        for i in mean_arr:
            line = '%f,\n' % i
            my_file.write(line)

    # FACE + BG
    p = os.path.join(P.LOG_BASE, f_bg + '.txt')
    tmp = np.genfromtxt(p, delimiter=',', dtype=float)
    mean_arr = np.mean(tmp, axis=1)

    p = os.path.join(P.LOG_BASE, mean_f_bg + '.txt')
    with open(p, 'a') as my_file:
        for i in mean_arr:
            line = '%f,\n' % i
            my_file.write(line)

    # FULL FRAMES
    p = os.path.join(P.LOG_BASE, ff + '.txt')
    tmp = np.genfromtxt(p, delimiter=',', dtype=float)
    mean_arr = np.mean(tmp, axis=1)

    p = os.path.join(P.LOG_BASE, mean_ff + '.txt')
    with open(p, 'a') as my_file:
        for i in mean_arr:
            line = '%f,\n' % i
            my_file.write(line)


# make_avg_preds()

def get_corr(name, target):
    path = os.path.join(P.LOG_BASE, name)
    predictions = np.genfromtxt(path, delimiter=',', dtype=float)
    predictions = predictions[:, 0]
    r, p = stats.pearsonr(predictions, target)

    print(name, 'r: ', r, 'p: ', p)


def correlation_avg_preds(num):
    target = D.basic_load_personality_labels('test')
    target = np.mean(target, axis=1)
    target = fix_labels(target)

    nname = 'pred_%d.txt' % num

    get_corr(nname, target)


correlation_avg_preds(182)

'''


179
180
181
182

'''





