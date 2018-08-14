import os
import numpy as np
import h5py
import deepimpression2.paths as P
import skvideo.io
from PIL import Image
import deepimpression2.chalearn10.align_crop as AC
from multiprocessing import Pool


def mp4_to_arr(video_path):
    vid = skvideo.io.vread(video_path)
    return vid


def convert(video_path):
    # for video_path in all_videos:
    video = mp4_to_arr(video_path)
    shape = video.shape
    face_pos = np.zeros((shape[0], 4), dtype=int)
    video_name = video_path.split('/')[-1].split('.mp4')[0] + '.h5'
    h5_path = os.path.join(P.CHALEARN30_ALL_DATA, video_name)

    print('..converting %s' % (video_name))

    with h5py.File(h5_path, 'w') as my_file:
        # for k in tqdm(range(shape[0])):
        for k in range(shape[0]):
            f = video[k]
            f = Image.fromarray(f, mode='RGB')
            f = f.resize((456, 256), Image.ANTIALIAS)  # size=(w, h)
            f = np.array(f)
            bb = AC.find_face_simple(f)
            if bb is None:
                if k == 0:
                    bb = [0, 0, 0, 0]
                else:
                    bb = face_pos[k-1]
            face_pos[k] = bb
            f = f.transpose([2, 0, 1])
            f = np.expand_dims(f, 0)
            my_file.create_dataset(name=str(k), data=f)

        my_file.create_dataset(name='faces', data=face_pos)


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


def parallel_convert(which, b, e, func, number_processes=20):
    all_videos = get_all_videos(which)
    pool = Pool(processes=number_processes)
    all_videos = all_videos[b:e]
    pool.apply_async(func)
    pool.map(func, all_videos)


def normal_convert(which, b, e):
    all_videos = get_all_videos(which)
    all_videos = all_videos[b:e]
    for video_path in all_videos:
        convert(video_path)


def check_converted():
    path = '/scratch/users/gabras/data/chalearn30/all_data/13kjwEtSyXc.003.h5'
    h5 = h5py.File(path, 'r')
    print(len(list(h5.keys())))
    print(len(h5['faces'][:]))
    h5.close()


# parallel_convert('test', 0, 500, convert)
# parallel_convert('test', 500, 1000, convert)
parallel_convert('test', 1000, 1500, convert)
# parallel_convert('test', 1500, 2000, convert)
