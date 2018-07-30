import h5py as h5
from deepimpression2.chalearn20.make_chalearn20_data import get_id_split
import deepimpression2.paths as P
import deepimpression2.constants as C1
import deepimpression2.chalearn20.constants as C2
from deepimpression2.chalearn20 import poisson_disc
from PIL import Image
import numpy as np
from random import randint
import os
import time
from random import shuffle


def visualize_grid(grid, points):
    grid = np.zeros((grid, grid, 3), dtype=np.uint8)
    for (x, y) in points:
        grid[x, y] = (255, 255, 255)
        grid[x + 1, y] = (255, 255, 255)
        grid[x, y + 1] = (255, 255, 255)
        grid[x + 1, y + 1] = (255, 255, 255)
        grid[x - 0, y] = (255, 255, 255)
        grid[x, y - 0] = (255, 255, 255)
        grid[x - 0, y - 0] = (255, 255, 255)

    img = Image.fromarray(grid, 'RGB')
    img.show()


def get_info_labels():
    numbers = get_id_split(only_numbers=True)
    train = h5.File(P.CHALEARN_TRAIN_LABELS_20, 'r')
    test = h5.File(P.CHALEARN_TEST_LABELS_20, 'r')
    val = h5.File(P.CHALEARN_VAL_LABELS_20, 'r')

    print('training videos: %d  unique ID: %d  video per UID: %f' % (len(train.keys()), numbers[0], float(len(train.keys())/float(numbers[0])) ) )
    print('testing videos: %d  unique ID: %d  video per UID: %f' % (len(test.keys()), numbers[1], float(len(test.keys())/float(numbers[1])) ) )
    print('validation videos: %d  unique ID: %d  video per UID: %f' % (len(val.keys()), numbers[2], float(len(val.keys())/float(numbers[2])) ) )

    train.close()
    test.close()
    val.close()


def get_batch_uid(which):
    # r is empirically selected for grid size and batch size
    # TODO: figure out relationship between r and grid and batch size
    assert(which in ['train', 'test', 'val'])

    if which == 'train':
        batch_size = C1.TRAIN_BATCH_SIZE
        grid = C2.NUM_TRAIN
        r = 290
    elif which == 'test':
        batch_size = C1.TEST_BATCH_SIZE
        grid = C2.NUM_TEST
        r = 70
    elif which == 'val':
        batch_size = C1.VAL_BATCH_SIZE
        grid = C2.NUM_VAL
        r = 70

    points = poisson_disc.poisson_disc_samples(grid, grid, r=r, k=batch_size)
    # print(len(points))

    if len(points) < batch_size:
        get_batch_uid(which)
    else:
        points = points[0:batch_size]

    # visualize_grid(grid, points)

    keys = get_id_split(which=which)
    x = [i[0] for i in points]
    y = [i[1] for i in points]

    uids_left = np.array(keys)[x]
    uids_right = np.array(keys)[y]

    return uids_left, uids_right


def get_keys(left_uids, right_uids, uid_keys_map):
    left_keys = []
    right_keys = []

    for l in left_uids:
        k = uid_keys_map[l][:]
        i = randint(0, len(k)-1)
        k = k[i].astype('str')
        left_keys.append(k)
    
    for r in right_uids:
        k = uid_keys_map[r][:]
        i = randint(0, len(k)-1)
        k = k[i].astype('str')
        right_keys.append(k)

    return left_keys, right_keys


def get_labels(labels_h5, left_keys, right_keys):
    # for each trait (left, right). example label: [0, 1, 1, 0, 0, 1, 1, 0, 1, 0] for OCEAS traits
    # (1, 0) = left
    # (0, 1) = right
    assert len(left_keys) == len(right_keys)
    tot = len(left_keys)

    all_one_hot_labels = np.zeros((tot, 10), dtype=np.uint8)

    def which_side(l, r):
        if l > r:
            return 1, 0
        else:
            return 0, 1

    for i in range(tot):
        left_label = labels_h5[left_keys[i]][:]
        right_label = labels_h5[right_keys[i]][:]
        one_hot_label = []
        for j in range(5):
            side = which_side(left_label[j], right_label[j])
            one_hot_label.append(side[0])
            one_hot_label.append(side[1])

        all_one_hot_labels[i] = one_hot_label

    return all_one_hot_labels


def get_frame(num_frames):
    num = randint(0, num_frames - 1)
    return num


def reading_test():
    video_names = os.listdir(P.DUMMY_DATA_JPG)

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


def all_data_reading():
    # left = np.zeros((2, 3, 208, 208), dtype=np.float32)
    l = os.listdir(P.CHALEARN_ALL_DATA_20_2)

    # shuffle(l)
    l = l[0:100]

    for i, v in enumerate(l):
        print('---')
        v_path = os.path.join(P.CHALEARN_ALL_DATA_20_2, v)
        with h5.File(v_path, 'r') as mf:
            # tot_frames = len(mf.keys())

            n = get_frame(len(mf.keys()))
            ts = time.time()
            img = mf[str(n)]
            print(time.time() - ts)
            # for i in range(tot_frames):
            #     img = mf[str(i)]


def quicker_load(k, id_frames):
    k = k.split('.mp4')[0]
    h5_path = os.path.join(P.CHALEARN_ALL_DATA_20_2, '%s.h5' % k)
    v = h5.File(h5_path, 'r')
    n = get_frame(id_frames[k][0])
    fe = v[str(n)][:]
    v.close()
    return fe


def get_data(left_keys, right_keys, id_frames):
    left = np.zeros((len(left_keys), 3, C2.SIDE, C2.SIDE), dtype=np.float32)
    right = np.zeros((len(right_keys), 3, C2.SIDE, C2.SIDE), dtype=np.float32)

    for i, k in enumerate(left_keys):
        left[i] = quicker_load(k, id_frames)

    for i, k in enumerate(right_keys):
        right[i] = quicker_load(k, id_frames)

    return left, right


def load_data(which, uid_keys_map, labs, id_frames):
    left_uids, right_uids = get_batch_uid(which)
    left_keys, right_keys = get_keys(left_uids, right_uids, uid_keys_map)
    labels = get_labels(labs, left_keys, right_keys)
    left_data, right_data = get_data(left_keys, right_keys, id_frames)

    return labels, left_data, right_data


def get_info_stefan_data():
    train_base = P.CHALEARN_FACES_TRAIN_H5
    val_base = P.CHALEARN_FACES_VAL_H5

    tot_train = 0

    f1 = os.listdir(train_base)
    for i in f1:
        f1_path = os.path.join(train_base, i)
        f2 = os.listdir(f1_path)
        for j in f2:
            f2_path = os.path.join(f1_path, j)
            f3 = os.listdir(f2_path)
            tot_train += len(f3)

    print('total train: %d' % tot_train)

    v = os.listdir(val_base)
    tot_val = len(v)
    print('total val: %d' % tot_val)


def get_info_chalearn20_data():
    train = h5.File(P.CHALEARN_TRAIN_DATA_20, 'r')
    test = h5.File(P.CHALEARN_TEST_DATA_20, 'r')
    val = h5.File(P.CHALEARN_VAL_DATA_20, 'r')

    print(len(train.keys()))
    print(len(test.keys()))
    print(len(val.keys()))

    print(len(train.keys()) + len(test.keys()) + len(val.keys()))

    train.close()
    test.close()
    val.close()


def num_frame_statistics():
    videos = os.listdir(P.CHALEARN_ALL_DATA_20_2)

    with h5.File(P.NUM_FRAMES, 'w') as mf:
        for i, v in enumerate(videos):
            print(i)
            p = os.path.join(P.CHALEARN_ALL_DATA_20_2, v)
            n = len(h5.File(p, 'r').keys())
            print('asdf')
            n = np.array([n], dtype=np.uint8)

            mf.create_dataset(name=v.split('.h5')[0], data=n)

    print('done')
