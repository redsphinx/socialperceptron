# TODO: utils for chalearn30 that depending on the mode only get face, bg or bg+face
import deepimpression2.paths as P
import os
import skvideo.io
import h5py as h5
from tqdm import tqdm
from deepimpression2.chalearn20 import data_utils as D
import numpy as np
from random import shuffle
import deepimpression2.chalearn20.constants as C2
from PIL import Image


def mp4_to_arr(video_path):
    vid = skvideo.io.vread(video_path)
    return vid


def get_all_videos(which):
    assert (which in ['train', 'test', 'val'])
    if which == 'test':
        top = P.CHALEARN_TEST_ORIGINAL
    elif which == 'val':
        top = P.CHALEARN_VAL_ORIGINAL
    elif which == 'train':
        top = P.CHALEARN_TRAIN_ORIGINAL

    all_videos = []

    l1 = os.listdir(top)  # train-1
    for i in l1:
        l1i = os.path.join(top, i)
        l2 = os.listdir(l1i)  # training80_01
        for j in l2:
            l2j = os.path.join(l1i, j)
            videos = os.listdir(l2j)  # asfdkj.mp4
            for v in videos:
                video_path = os.path.join(l2j, v)
                all_videos.append(video_path)

    all_videos.sort()
    return all_videos


def check_which_not_done(which, b, e):
    all_videos = get_all_videos(which)[b:e]
    save_path = '/scratch/users/gabras/data/chalearn30/todo_%s.txt' % (which)

    for vp in tqdm(all_videos):
        v = mp4_to_arr(vp)
        frames_video = v.shape[0]
        video_name = vp.split('/')[-1].split('.mp4')[0] + '.h5'
        h5_path = os.path.join(P.CHALEARN30_ALL_DATA, video_name)
        with h5.File(h5_path, 'r') as my_file:
            frames_h5 = len(my_file.keys()) - 1

        if frames_video != frames_h5:
            with open(save_path, 'a') as todo:
                todo.write('%s/n' % (vp))
            print(vp)


def only_names_check_which_not_done(which, b, e):
    all_videos = get_all_videos(which)[b:e]

    for i, vp in enumerate(all_videos):
        # for i, vp in tqdm(enumerate(all_videos)):
        video_name = vp.split('/')[-1].split('.mp4')[0] + '.h5'
        h5_path = os.path.join(P.CHALEARN30_ALL_DATA, video_name)
        if not os.path.exists(h5_path):
            print('asdf')
            print(i, vp)

# -----------------------------------
# for loading data
# -----------------------------------


def quicker_load(k, id_frames, which_data):
    k = k.split('.mp4')[0]
    h5_path = os.path.join(P.CHALEARN30_ALL_DATA, '%s.h5' % k)
    v = h5.File(h5_path, 'r')

    n = D.get_frame(id_frames[k][0])
    try:
        fe = v[str(n)][:]  # shape=(1, c, h, w)
    except KeyError:
        print('KeyError: %d does not exist in %s' % (n, k))
        real_len = len(v.keys()) - 1
        print('total len of h5 file is %d' % (real_len))
        n = D.get_frame(real_len)
        fe = v[str(n)][:]

    if which_data in ['face', 'bg']:
        optface = v['faces'][n]
    else:
        optface = None

    v.close()
    return fe, optface


def fill_average(image, which_data, optface):
    if optface is not None:
        if optface[3] > C2.H:
            optface[3] = C2.H
        if optface[2] > C2.W:
            optface[2] = C2.W
        if optface[1] < 0:
            optface[1] = 0
        if optface[0] < 0:
            optface[0] = 0

    # print(optface)

    if which_data == 'all':
        if optface is None:
            return image
        else:
            print('Problem: which_data == all but optface not None')
            return None

    else:
        h, w = image.shape[2], image.shape[3]  # data is transposed before save

        if which_data == 'bg':
            px_mean = np.mean(image, 2)
            px_mean = np.mean(px_mean, 2)

            for i in range(optface[0], optface[2]):
                for j in range(optface[1], optface[3]):
                    try:
                        image[:, :, j, i] = px_mean
                    except IndexError:
                        print(2, IndexError, j, i)

            image = image.astype(np.uint8)

            # img1 = Image.fromarray(np.transpose(image[0], (1, 2, 0)), mode='RGB')
            # p = '/home/gabras/deployed/deepimpression2/chalearn30/bg'
            # img1.save('%s/one.jpg' % p)

            return image

        elif which_data == 'face':
            tot = 0
            px_mean = np.zeros((1, 3))
            # print(optface)

            for i in range(optface[0], optface[2]):  # w
                for j in range(optface[1], optface[3]):  # h
                    try:
                        px_mean += image[:, :, j, i]
                    except IndexError:
                        print(3, IndexError, j, i, optface)
                    tot += 1

            if tot == 0:
                tot = 1
            px_mean /= tot

            px_mean = np.expand_dims(px_mean, -1)
            px_mean = np.expand_dims(px_mean, -1)

            new_bg = np.ones(image.shape) * px_mean
            for i in range(optface[0], optface[2]):
                for j in range(optface[1], optface[3]):
                    try:
                        new_bg[:, :, j, i] = image[:, :, j, i]
                    except IndexError:
                        print(4, IndexError, j, i, optface)

            image = new_bg
            image = image.astype(np.uint8)

            # img1 = Image.fromarray(np.transpose(image[0], (1, 2, 0)), mode='RGB')
            # p = '/home/gabras/deployed/deepimpression2/chalearn30/face'
            # img1.save('%s/one.jpg' % p)

            return image


def get_data(keys, id_frames, which_data):
    data = np.zeros((len(keys), 3, C2.H, C2.W), dtype=np.float32)

    for i, k in enumerate(keys):
        image, optface = quicker_load(k, id_frames, which_data)
        image = fill_average(image, which_data, optface)
        data[i] = image

    return data


def load_data(labs_selected, labs_h5, id_frames, which_data):
    assert(which_data in ['bg', 'face', 'all'])

    labels = np.zeros((len(labs_selected), 5), dtype=np.float32)

    shuffle(labs_selected)
    keys = []
    for i in range(len(labs_selected)):
        k = labs_selected[i]
        keys.append(k)
        labels[i] = labs_h5[k][0:5]

    data = get_data(keys, id_frames, which_data)
    return labels, data


def check_saved_faces():
    val_labels = h5.File(P.CHALEARN_VAL_LABELS_20, 'r')
    _labs = list(val_labels)
    id_frames = h5.File(P.NUM_FRAMES, 'r')

    labels = np.zeros((len(_labs), 5), dtype=np.float32)
    which_data = 'face'

    shuffle(_labs)
    keys = []
    for i in range(len(_labs)):
        k = _labs[i]
        keys.append(k)
        labels[i] = val_labels[k][0:5]

    # data = np.zeros((len(keys), 3, C2.H, C2.W), dtype=np.float32)

    for i, k in enumerate(keys):
        image, optface = quicker_load(k, id_frames, which_data)
        if optface[1] > 456 or optface[3] > 256:
            print(k.split('.mp4')[0], optface)
            image = np.transpose(image[0], (1, 2, 0))
            img = Image.fromarray(image, mode='RGB')
            p = '/home/gabras/deployed/deepimpression2/chalearn30/wrongs'
            img.save('%s/%s.jpg' %(p, k.split('.mp4')[0]))

            # !! turns out that the face rectangle has to be a square so if the face is near bottom of screen, the
            # square will spill over the height limit

# def find_best_val():
#     