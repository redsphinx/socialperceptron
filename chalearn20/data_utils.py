import h5py as h5
from deepimpression2.chalearn20.make_chalearn20_data import get_id_split
import deepimpression2.chalearn20.paths as P
import os
import deepimpression2.constants as C1
import deepimpression2.chalearn20.constants as C2
from deepimpression2.chalearn20 import poisson_disc
from PIL import Image
import numpy as np


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


def get_train_batch_uid():
    batch_size = C1.TRAIN_BATCH_SIZE
    grid = C2.NUM_TRAIN

    points = poisson_disc.poisson_disc_samples(grid, grid, r=290, k=batch_size)
    # print(len(points))

    if len(points) < batch_size:
        get_train_batch_uid()
    elif len(points) > batch_size:
        points = points[0:batch_size]
    else:
        pass

    # visualize_grid(grid, points)

    keys = get_id_split(which='train')
    x = [i[0] for i in points]
    y = [i[1] for i in points]

    uids_left = np.array(keys)[x]
    uids_right = np.array(keys)[y]

    return uids_left, uids_right


get_train_batch_uid()




def get_labels():
    # TODO: one-hot encode based on pair
    # TODO: make for training + validation
    pass


