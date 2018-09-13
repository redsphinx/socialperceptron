# utils for chalearn30
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
import chainer


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


def quicker_load(k, id_frames, which_data, ordered=False):
    k = k.split('.mp4')[0]
    h5_path = os.path.join(P.CHALEARN30_ALL_DATA, '%s.h5' % k)
    v = h5.File(h5_path, 'r')

    n, zero_frames = D.get_frame(id_frames[k][0], ordered)

    if zero_frames:
        with open(P.ZERO_FRAMES, 'a') as my_file:
            my_file.write('%s\n' % v)

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


def quicker_load_resize(k, id_frames, which_data, ordered=False):
    k = k.split('.mp4')[0]

    if which_data == 'face':
        h5_path = os.path.join(P.CHALEARN_ALL_DATA_20_2, '%s.h5' % k)
    else:
        h5_path = os.path.join(P.CHALEARN30_ALL_DATA, '%s.h5' % k)

    v = h5.File(h5_path, 'r')

    n, zero_frames = D.get_frame(id_frames[k][0], ordered)

    if zero_frames:
        with open(P.ZERO_FRAMES, 'a') as my_file:
            my_file.write('%s\n' % v)

    try:
        fe = v[str(n)][:]  # shape=(1, c, h, w)
    except KeyError:
        print('KeyError: %d does not exist in %s' % (n, k))
        real_len = len(v.keys()) - 1
        print('total len of h5 file is %d' % (real_len))
        n, zero_frames = D.get_frame(real_len)
        fe = v[str(n)][:]

    if which_data == 'bg':
        optface = v['faces'][n]
    else:
        optface = None

    v.close()
    return fe, optface


def fill_average(image, which_data, optface, resize=False):
    if optface is not None: # bg (or face if resize == false)
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
            if resize:  # crop
                image = np.transpose(image[0], (1, 2, 0))
                img = Image.fromarray(image, mode='RGB')
                img = img.crop((100, 0, 356, 256))  # left, upper, right, and lower
                # save image to see if good
                # img.save('/home/gabras/deployed/deepimpression2/chalearn30/crops/crop_all.jpg')
                img = np.array(img)
                image = np.transpose(img, (2, 0, 1))
                image = np.expand_dims(image, 0)

            return image
        else:
            print('Problem: which_data == all but optface not None')
            return None

    else:
        # h, w = image.shape[2], image.shape[3]  # data is transposed before save

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

            if resize:
                if optface[0] > (456 - optface[2]):
                    left = 0
                    right = 256
                else:
                    left = 200
                    right = 456

                image = np.transpose(image[0], (1, 2, 0))
                img = Image.fromarray(image, mode='RGB')
                img = img.crop((left, 0, right, 256))  # left, upper, right, and lower
                # save image to see if good
                # img.save('/home/gabras/deployed/deepimpression2/chalearn30/crops/crop_bg.jpg')
                img = np.array(img)
                image = np.transpose(img, (2, 0, 1))
                image = np.expand_dims(image, 0)

            # img1 = Image.fromarray(np.transpose(image[0], (1, 2, 0)), mode='RGB')
            # p = '/home/gabras/deployed/deepimpression2/chalearn30/bg'
            # img1.save('%s/one.jpg' % p)

            return image

        elif which_data == 'face':
            if resize:
                image = np.transpose(image, (1, 2, 0))
                img = Image.fromarray(image, mode='RGB')
                img = img.resize((C2.RESIDE, C2.RESIDE))
                # save image to see if good
                # img.save('/home/gabras/deployed/deepimpression2/chalearn30/crops/crop_face.jpg')
                img = np.array(img)
                image = np.transpose(img, (2, 0, 1))
                image = np.expand_dims(image, 0)

                return image

            else:
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


def get_data(keys, id_frames, which_data, resize=False, ordered=False, twostream=False):
    if resize:
        if twostream:
            data = np.zeros((len(keys), 6, C2.RESIDE, C2.RESIDE), dtype=np.float32)
        else:
            data = np.zeros((len(keys), 3, C2.RESIDE, C2.RESIDE), dtype=np.float32)
    else:
        data = np.zeros((len(keys), 3, C2.H, C2.W), dtype=np.float32)

    if resize:
        if twostream:
            if which_data == 'all':
                for i, k in enumerate(keys):
                    bg, optface = quicker_load_resize(k, id_frames, 'bg', ordered)
                    bg = fill_average(bg, 'bg', optface, resize)
                    face, optface = quicker_load_resize(k, id_frames, 'face', ordered)
                    face = fill_average(face, 'face', optface, resize)
                    data[i] = np.concatenate((bg, face), axis=1)
        else:
            for i, k in enumerate(keys):
                image, optface = quicker_load_resize(k, id_frames, which_data, ordered)
                image = fill_average(image, which_data, optface, resize)
                data[i] = image

    else:
        for i, k in enumerate(keys):
            image, optface = quicker_load(k, id_frames, which_data, ordered)
            image = fill_average(image, which_data, optface)
            data[i] = image

    return data


def load_data(labs_selected, labs_h5, id_frames, which_data, resize=False, ordered=False, twostream=False):
    assert(which_data in ['bg', 'face', 'all'])

    labels = np.zeros((len(labs_selected), 5), dtype=np.float32)

    if not ordered:
        shuffle(labs_selected)
    keys = []
    for i in range(len(labs_selected)):
        k = labs_selected[i]
        keys.append(k)
        labels[i] = labs_h5[k][0:5]

    data = get_data(keys, id_frames, which_data, resize, ordered, twostream)
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


def find_best_val():
    log14 = P.LOG_BASE + 'val_%d.txt' % (53)
    # log33 = P.LOG_BASE + 'val_%d.txt' % (33)
    # log34 = P.LOG_BASE + 'val_%d.txt' % (34)

    logs = [log14] #, log33, log34]

    for l in logs:
        print(l)
        best = np.genfromtxt(l, 'float', delimiter=',')
        best = np.transpose(best, (1, 0))[0]
        # get loss every 10 epochs
        tmp_best = []
        r = []
        for e in range(0, 110, 10):
            i = e
            if e != 0:
                i -= 1
                r.append(i)
                tmp_best.append(best[i])

        tmp_best = np.asarray(tmp_best)
        print(tmp_best)
        print('worst = ', r[np.argmax(tmp_best, axis=0)])
        print('best = ', r[np.argmin(tmp_best, axis=0)])


# find_best_val()


def load_model(model, path_to_weights, load_weights=False):
    if load_weights:
        chainer.serializers.load_npz(path_to_weights, model)

    return model


def load_last_layers(model_to, model_from, load_weights=False):
    my_model = model_to

    if load_weights:
        fc_b = model_from.__getattribute__('fc').b
        fc_w = model_from.__getattribute__('fc').W

        my_model.__getattribute__('fc').__setattr__('b', fc_b)
        my_model.__getattribute__('fc').__setattr__('W', fc_w)

    return my_model


def basic_load_personality_labels(which):
    assert(which in ['test', 'train', 'val'])

    if which == 'test':
        labels = h5.File(P.CHALEARN_TEST_LABELS_20, 'r')
    elif which == 'train':
        labels = h5.File(P.CHALEARN_TRAIN_LABELS_20, 'r')
    else:
        labels = h5.File(P.CHALEARN_VAL_LABELS_20, 'r')

    final_labels = np.zeros((len(list(labels)), 5), dtype=np.float32)

    for i, k in enumerate(list(labels)):
        final_labels[i] = labels[k][:][:5]

    return final_labels