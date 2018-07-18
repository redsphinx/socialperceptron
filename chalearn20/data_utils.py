import h5py as h5
from deepimpression2.chalearn20.make_chalearn20_data import get_id_split
import deepimpression2.chalearn20.paths as P
import os


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


def safe_mkdir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

get_info_labels()