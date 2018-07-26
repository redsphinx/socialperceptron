# create files to make clean identity split data
# create labels accordingly
import deepimpression2.chalearn20.paths as P
import deepimpression2.paths as P2
import deepimpression2.chalearn20.constants as C
import numpy as np
import pickle as pkl
import os
import h5py as h5
from tqdm import tqdm


def get_trait_labels():
    encoding = 'latin1'

    f_test = open(P.CHALEARN_TEST_LABELS_ORIGINAL, 'rb')
    f_val = open(P.CHALEARN_VAL_LABELS_ORIGINAL, 'rb')
    f_train = open(P.CHALEARN_TRAIN_LABELS_ORIGINAL, 'rb')
    annotation_test = pkl.load(f_test, encoding=encoding)
    annotation_val = pkl.load(f_val, encoding=encoding)
    annotation_train = pkl.load(f_train, encoding=encoding)

    return annotation_train, annotation_val, annotation_test


def get_geth_labels():
    geth_test = np.genfromtxt(P.CHALEARN_TEST_GETH_LABELS_ORIGINAL, str, delimiter=';')
    geth_test = geth_test.tolist()[1:]
    geth_dev = np.genfromtxt(P.CHALEARN_DEV_GETH_LABELS_ORIGINAL, str, delimiter=';')
    geth_dev = geth_dev.tolist()[1:]
    return geth_dev, geth_test


def get_unique_ids():
    # get all the names
    annotation_train, annotation_val, annotation_test = get_trait_labels()

    # ['extraversion', 'neuroticism', 'agreeableness', 'conscientiousness', 'interview', 'openness']
    annotation_test_keys = list(annotation_test.keys())
    annotation_val_keys = list(annotation_val.keys())
    annotation_train_keys = list(annotation_train.keys())

    test_video_names = annotation_test[annotation_test_keys[0]].keys()
    val_video_names = annotation_val[annotation_val_keys[0]].keys()
    train_video_names = annotation_train[annotation_train_keys[0]].keys()

    # get the unique names from splits
    un_test = []
    for i, v in enumerate(test_video_names):
        name = v.split('.')[0]
        if name not in un_test:
            un_test.append(name)
    un_val = []
    for i, v in enumerate(val_video_names):
        name = v.split('.')[0]
        if name not in un_val:
            un_val.append(name)
    un_train = []
    for i, v in enumerate(train_video_names):
        name = v.split('.')[0]
        if name not in un_train:
            un_train.append(name)

    all_uniques = list(set(un_train + un_test + un_val))
    return all_uniques


def get_id_split(which='all', only_numbers=False):
    if only_numbers:
        if 'all':
            return [C.NUM_TRAIN, C.NUM_TEST, C.NUM_VAL]
        elif 'train':
            return C.NUM_TRAIN
        elif 'test':
            return C.NUM_TEST
        elif 'val':
            return C.NUM_VAL

    if not os.path.exists(P.CHALEARN_TRAIN_SPLIT):
        # create the data splits
        uids = get_unique_ids()
        train_split = uids[0:C.NUM_TRAIN]
        val_split = uids[C.NUM_TRAIN:C.NUM_TRAIN + C.NUM_VAL]
        test_split = uids[C.NUM_TRAIN + C.NUM_VAL:C.NUM_TRAIN + C.NUM_VAL + C.NUM_TEST]

        # write to file
        with open(P.CHALEARN_TRAIN_SPLIT, 'a') as f:
            for i in train_split:
                line = '%s\n' % i
                f.write(line)

        with open(P.CHALEARN_VAL_SPLIT, 'a') as f:
            for i in val_split:
                line = '%s\n' % i
                f.write(line)

        with open(P.CHALEARN_TEST_SPLIT, 'a') as f:
            for i in test_split:
                line = '%s\n' % i
                f.write(line)

        get_id_split()
    else:
        if which == 'all':
            train_split = list(np.genfromtxt(P.CHALEARN_TRAIN_SPLIT, str))
            val_split = list(np.genfromtxt(P.CHALEARN_VAL_SPLIT, str))
            test_split = list(np.genfromtxt(P.CHALEARN_TEST_SPLIT, str))
            return train_split, val_split, test_split
        elif which == 'train':
            train_split = list(np.genfromtxt(P.CHALEARN_TRAIN_SPLIT, str))
            return train_split
        elif which == 'test':
            test_split = list(np.genfromtxt(P.CHALEARN_TEST_SPLIT, str))
            return test_split
        elif which == 'val':
            val_split = list(np.genfromtxt(P.CHALEARN_VAL_SPLIT, str))
            return val_split


def create_labels_data():
    train_split, val_split, test_split = get_id_split()

    annotation_train, annotation_val, annotation_test = get_trait_labels()
    geth_dev, geth_test = get_geth_labels()

    def transform1(annotation):
        annotation_keys = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism', 'interview']
        names = list(annotation[annotation_keys[0]].keys())
        all_traits = []
        for n in names:
            traits = []
            for v in annotation_keys:
                tr = annotation[v][n]
                traits.append(tr)

            all_traits.append(traits)

        return names, all_traits

    names_train, traits_train = transform1(annotation_train)
    names_val, traits_val = transform1(annotation_val)
    names_test, traits_test = transform1(annotation_test)

    all_names_trait = names_train + names_val + names_test
    all_traits = traits_train + traits_val + traits_test
    dict_traits = dict(zip(all_names_trait, all_traits))
    del names_train, names_val, names_test, traits_train, traits_val, traits_test, all_traits

    def transform2(geth):
        names = []
        all_geth = []
        for i in geth:
            names.append(i[0])
            geth = [float(i[3]), float(i[2])]
            all_geth.append(geth)
        return names, all_geth

    names_geth_test, traits_geth_test = transform2(geth_test)
    names_geth_dev, traits_geth_dev = transform2(geth_dev)

    all_names_geth = names_geth_test + names_geth_dev
    all_geth = traits_geth_test + traits_geth_dev
    dict_geth = dict(zip(all_names_geth, all_geth))
    del names_geth_test, names_geth_dev, traits_geth_test, traits_geth_dev, all_geth

    def make_h5_label(data_split, annotation, geth, all_names, h5_path):
        action = 'a' if os.path.exists(h5_path) else 'w'

        with h5.File(h5_path, action) as f:
            for i in data_split:
                names = [all_names[k] for k, v in enumerate(all_names) if i in v]
                for n in names:
                    v = annotation[n]
                    v.extend(geth[n])
                    f.create_dataset(name=n, data=v)

    # make_h5_label(train_split, dict_traits, dict_geth, all_names_trait, 'thing.h5')

    # make_h5_label(train_split, dict_traits, dict_geth, all_names_trait, P.CHALEARN_TRAIN_LABELS_20)
    # make_h5_label(val_split, dict_traits, dict_geth, all_names_trait, P.CHALEARN_VAL_LABELS_20)
    # make_h5_label(test_split, dict_traits, dict_geth, all_names_trait, P.CHALEARN_TEST_LABELS_20)

# create_labels_data()


def create_uid_keys_mapping():
    trl = h5.File(P.CHALEARN_TRAIN_LABELS_20, 'r')
    tel = h5.File(P.CHALEARN_TEST_LABELS_20, 'r')
    vl = h5.File(P.CHALEARN_VAL_LABELS_20, 'r')

    trs, vs, tes = get_id_split()

    def make_map(l, sp, h5p):

        ke = list(l.keys())
        with h5.File(h5p, 'w') as f:
            for s in sp:
                ks_tmp = []
                for k in ke:
                    if s in k:
                        ks_tmp.append(k)

                ks_tmp = np.array(ks_tmp, dtype='S')
                if ks_tmp.shape[0] == 0:
                    print('asdf')
                f.create_dataset(name=s, data=ks_tmp)

    make_map(trl, trs, P2.TRAIN_UID_KEYS_MAPPING)
    make_map(tel, tes, P2.TEST_UID_KEYS_MAPPING)
    make_map(vl, vs, P2.VAL_UID_KEYS_MAPPING)

    trl.close()
    tel.close()
    vl.close()


def create_chalearn_data(which):
    # fix stefan data
    if not os.path.exists(P.CHALEARN_TRAIN_DATA_20):
        action = 'w'
    else:
        action = 'a'

    # action = 'r'  # for debugging

    train_h5 = h5.File(P.CHALEARN_TRAIN_DATA_20, action)
    # train_h5.close()
    val_h5 = h5.File(P.CHALEARN_VAL_DATA_20, action)
    # val_h5.close()
    test_h5 = h5.File(P.CHALEARN_TEST_DATA_20, action)
    # test_h5.close()

    train_labels = h5.File(P2.CHALEARN_TRAIN_LABELS_20, 'r')
    test_labels = h5.File(P2.CHALEARN_TEST_LABELS_20, 'r')
    val_labels = h5.File(P2.CHALEARN_VAL_LABELS_20, 'r')

    # --

    def check(i, f2p, labels, h5_data):
        # keys with .mp4
        imp4 = i + '.mp4'
        if imp4 in list(labels.keys()):
            if imp4 not in list(h5_data.keys()):
                v_path = os.path.join(f2p, i + '.h5')
                h5_file = h5.File(v_path, 'r')
                # keys: metadata, video, audio
                h5_file_video = h5_file['video'][:]
                h5_data.create_dataset(name=imp4, data=h5_file_video)
                # print('allocated')
                return True
            else:
                return False
        else:
            return False

    # TODO: get test

    if which == 'train':
        train = os.listdir(P2.CHALEARN_FACES_TRAIN_H5)
        train.sort()
        train.pop()  # remove empty validation folder
        for f1 in train:
            f1_path = os.path.join(P2.CHALEARN_FACES_TRAIN_H5, f1)
            divs = os.listdir(f1_path)
            for f2 in divs:
                f2_path = os.path.join(f1_path, f2)
                videos = os.listdir(f2_path)
                for v in videos:
                    # is v in the split
                    v = v.split('.h5')[0]
                    # print(v)
                    check(v, f2_path, train_labels, train_h5)
                    check(v, f2_path, test_labels, test_h5)
                    check(v, f2_path, val_labels, val_h5)

    elif which == 'val':
        val = os.listdir(P2.CHALEARN_FACES_VAL_H5)
        val.sort()
        # val = [val[-1]]  # for adding the one we popped
        for v in val:
            v = v.split('.h5')[0]
            allocated1 = check(v, P2.CHALEARN_FACES_VAL_H5, train_labels, train_h5)
            allocated2 = check(v, P2.CHALEARN_FACES_VAL_H5, test_labels, test_h5)
            allocated3 = check(v, P2.CHALEARN_FACES_VAL_H5, val_labels, val_h5)
            if (int(allocated1) + int(allocated2) + int(allocated3)) == 0:
                print('not allocated: %s' % v)

    elif which == 'test':
        test = os.listdir(P2.CHALEARN_FACES_TEST_H5)
        test.sort()
        # val = [val[-1]]  # for adding the one we popped
        for v in test:
            v = v.split('.h5')[0]
            allocated1 = check(v, P2.CHALEARN_FACES_VAL_H5, train_labels, train_h5)
            allocated2 = check(v, P2.CHALEARN_FACES_VAL_H5, test_labels, test_h5)
            allocated3 = check(v, P2.CHALEARN_FACES_VAL_H5, val_labels, val_h5)
            if (int(allocated1) + int(allocated2) + int(allocated3)) == 0:
                print('not allocated: %s' % v)

    train_h5.close()
    test_h5.close()
    val_h5.close()
    train_labels.close()
    test_labels.close()
    val_labels.close()

    print('done with %s' % which)


# create_chalearn_data('val')
