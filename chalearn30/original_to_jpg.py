import os
import numpy as np
import h5py
import deepimpression2.paths as P
import skvideo.io
from PIL import Image
import deepimpression2.chalearn10.align_crop as AC


def mp4_to_arr(video_path):
    vid = skvideo.io.vread(video_path)
    return vid


def convert(which):
    assert(which in ['train', 'test', 'val'])
    if which == 'test':
        top = P.CHALEARN_TEST_ORIGINAL
    elif which == 'val':
        top = P.CHALEARN_VAL_ORIGINAL
    elif which == 'train':
        top = P.CHALEARN_TRAIN_ORIGINAL

    l1 = os.listdir(top)  # train-1
    for i in l1:
        l1i = os.path.join(top, i)
        l2 = os.listdir(l1i)  # training80_01
        for j in l2:
            l2j = os.path.join(l1i, j)
            videos = os.listdir(l2j)  # asfdkj.mp4
            for v in videos:
                video_path = os.path.join(l2j, v)
                video = mp4_to_arr(video_path)
                shape = video.shape
                face_pos = np.zeros((shape[0], 4), dtype=int)
                h5_path = os.path.join(P.CHALEARN30_ALL_DATA, v.split('.mp4')[0] + '.h5')

                with h5py.File(h5_path, 'w') as my_file:
                    for k in range(shape[0]):
                        f = video[k]
                        f = Image.fromarray(f, mode='RGB')
                        f = f.resize((456, 256), Image.ANTIALIAS)  # size=(w, h)
                        f = np.array(f)
                        bb = AC.find_face_simple(f)
                        face_pos[k] = bb
                        f = f.transpose([2, 0, 1])
                        f = np.expand_dims(f, 0)
                        my_file.create_dataset(name=str(k), data=f)

                    my_file.create_dataset(name='faces', data=face_pos)
                    # TODO: make parallel


convert('test')
