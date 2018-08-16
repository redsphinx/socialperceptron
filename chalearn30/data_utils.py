# TODO: utils for chalearn30 that depending on the mode only get face, bg or bg+face
import deepimpression2.paths as P
import os
import skvideo.io
import h5py
from tqdm import tqdm


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
    # for i, vp in tqdm(enumerate(all_videos)):
        v = mp4_to_arr(vp)
        frames_video = v.shape[0]
        video_name = vp.split('/')[-1].split('.mp4')[0] + '.h5'
        h5_path = os.path.join(P.CHALEARN30_ALL_DATA, video_name)
        with h5py.File(h5_path, 'r') as my_file:
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



# check_which_not_done('test', 1400) # fails from 1400 + 524
# check_which_not_done('train', 0, 500) # fails from 425
# check_which_not_done('train', 500, 1000) # fails from 500 + 126
# only_names_check_which_not_done('train', 626, 1000)
