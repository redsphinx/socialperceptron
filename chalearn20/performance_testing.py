# comparing different data storage solutions
# SETUP 1
# data
#   |_ID_0
#       |_v0.h5
#       |_v1.h5
#       ...
#   |_ID_1
#   ...
#
# vs. SETUP 2 -- doesn't make sense anymore
#
# data
#   |_ID_0.h5
#       k=v0, val
#
# vs. SETUP 3
#
# data.h5
#   k=v0, val
#
# vs. SETUP 4
#
# jpgs
# goal 1: find out if loading 1 by 1 from h5 is as fast as batch loading
# goal 2: find out if loading 1 by 1 from h5 is faster than loading jpg 1 by 1

import numpy as np
import h5py as h5
import deepimpression2.paths as P
import os
from deepimpression2.util import safe_mkdir
from scipy import ndimage
import time
import pickle as pkl


def make_setup1():
    # make h5 for each video
    video_names = os.listdir(P.DUMMY_DATA_JPG)
    safe_mkdir(P.SETUP1)

    for n in video_names:
        p = os.path.join(P.DUMMY_DATA_JPG, n)
        ph5 = os.path.join(P.SETUP1, '%s' % (n + '.h5'))

        frames = os.listdir(p)
        with h5.File(ph5, 'w') as h5_file:
            for f in frames:
                if not f.split('.')[-1] == 'wav':
                    fp = os.path.join(p, f)
                    img_arr = ndimage.imread(fp)
                    h5_file.create_dataset(name=f, data=img_arr)


def make_setup3():
    # make one h5 for all videos
    video_names = os.listdir(P.DUMMY_DATA_JPG)
    safe_mkdir(P.SETUP3)
    one_h5 = os.path.join(P.SETUP3, 'one.h5')

    with h5.File(one_h5, 'w') as h5_file:
        for n in video_names:

            p = os.path.join(P.DUMMY_DATA_JPG, n)
            frames = os.listdir(p)
            frames.sort() # order of frames matter
            frames.pop() # rm wav

            # frames, side, side, channels
            all_frames_arr = np.zeros((len(frames), 208, 208, 3), dtype=np.uint8)

            for i, f in enumerate(frames):
                if not f.split('.')[-1] == 'wav':
                    fp = os.path.join(p, f)
                    all_frames_arr[i] = ndimage.imread(fp)

            h5_file.create_dataset(name=n, data=all_frames_arr)


def make_setup4():
    # make one h5 for all videos
    video_names = os.listdir(P.DUMMY_DATA_JPG)
    safe_mkdir(P.SETUP4)

    pkl_name = os.path.join(P.SETUP4, 'one.pkl')

    all_frames_dict = {}

    for n in video_names:

        p = os.path.join(P.DUMMY_DATA_JPG, n)
        frames = os.listdir(p)
        frames.sort()  # order of frames matter
        frames.pop()  # rm wav

        # frames, side, side, channels
        all_frames_arr = np.zeros((len(frames), 208, 208, 3), dtype=np.uint8)

        for i, f in enumerate(frames):
            if not f.split('.')[-1] == 'wav':
                fp = os.path.join(p, f)
                all_frames_arr[i] = ndimage.imread(fp)

        # to list, can easily be converted into ndarray with np.array
        all_frames_arr = all_frames_arr.tolist()
        # to dir
        all_frames_dict[n] = all_frames_arr

    with open(pkl_name, 'wb') as my_pkl:
        pkl.dump(all_frames_dict, my_pkl)


def reading_test():
    video_names = os.listdir(P.DUMMY_DATA_JPG)
    total_f = 0
    total_n = len(video_names)

    # time_start = time.time()
    # for n in video_names:
    #     p = os.path.join(P.DUMMY_DATA_JPG, n)
    #     frames = os.listdir(p)
    #     frames.sort()  # order of frames matter
    #     frames.pop()  # rm wav
    #     total_f += len(frames)
    #     for f in frames:
    #         fp = os.path.join(p, f)
    #         tmp = ndimage.imread(fp)
    #
    # time_jpg = (time.time() - time_start)
    # print(total_f, total_n)
    # print('time jpg: %s' % str(time_jpg))
    #
    time_start = time.time()
    for n in video_names:
        name_h5 = os.path.join(P.SETUP1, '%s.h5' % (n))
        h5_file = h5.File(name_h5, 'r')
        keys = h5_file.keys()
        for k in keys:
            # print(k)
            # ts = time.time()
            img = h5_file[k][:]
            # print(time.time() - ts)
        h5_file.close()

    time_setup1 = (time.time() - time_start)
    print('time setup1: %s' % str(time_setup1))
    #
    # time_start = time.time()
    # name_h5 = os.path.join(P.SETUP3, 'one.h5')
    # h5_file = h5.File(name_h5, 'r')
    #
    # for n in video_names:
    #     video = h5_file[n][:]
    #     for i in range(video.shape[0]):
    #         img = video[i]
    #
    # time_setup3 = (time.time() - time_start)
    # print('time setup3: %s' % str(time_setup3))
    #
    # time_start = time.time()
    # name_h5 = os.path.join(P.SETUP3, 'one.h5')
    # h5_file = h5.File(name_h5, 'r')
    #
    # for n in video_names:
    #     for i in range(h5_file[n].shape[0]):
    #         img = h5_file[n][i]
    #
    # time_setup3_2 = (time.time() - time_start)
    # # print('time setup3_2: %s' % str(time_setup3_2))
    #
    # return [time_jpg, time_setup1, time_setup3, time_setup3_2]

    # time_start = time.time()
    # encoding = 'latin1'
    # name_pkl = os.path.join(P.SETUP4, 'one.pkl')
    #
    # my_pkl = open(name_pkl, 'rb')
    # pf = pkl.load(my_pkl, encoding=encoding)
    #
    # for n in video_names:
    #     nums = len(pf[n])
    #     for f in range(nums):
    #         img = pf[n][f]
    #
    # time_setup4 = (time.time() - time_start)
    #
    return time_setup1

    # return [time_jpg, time_setup1, time_setup3, time_setup3_2]


# num = 1
# t = np.zeros((num, 1))
# for i in range(num):
#     t[i] = reading_test()
#
# print(t)
# print(t.mean(axis=0))


def all_data_reading():
    l = os.listdir(P.CHALEARN_ALL_DATA_20_2)
    for v in l:
        v_path = os.path.join(P.CHALEARN_ALL_DATA_20_2, v)
        with h5.File(v_path, 'r') as mf:
            # tot_frames = len(mf.keys())
            ts = time.time()
            img = mf[str(0)]
            # for i in range(tot_frames):
            #     img = mf[str(i)]
            print(time.time() - ts)


# all_data_reading()


'''
space on disk

setup1:     696M
setup3:     694M
jpg:        76M
pkl:        3.2G

reading speed

setup1:     1.02 s
setup3:     0.25 s      
setup3_2:   0.98 s
jpg:        5.55 s
pkl:        116 s


'''
