import os
import numpy as np
import h5py
import deepimpression2.paths as P
from PIL import Image
import deepimpression2.chalearn10.align_crop as AC
from multiprocessing import Pool
import deepimpression2.chalearn30.data_utils as DU
import shutil
from tqdm import tqdm


def convert(video_path):
    # for video_path in all_videos:
    video = DU.mp4_to_arr(video_path)
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


def parallel_convert(which, b, e, func, number_processes=20):
    all_videos = DU.get_all_videos(which)
    pool = Pool(processes=number_processes)
    all_videos = all_videos[b:e]
    pool.apply_async(func)
    pool.map(func, all_videos)


def normal_convert(which, b, e):
    all_videos = DU.get_all_videos(which)
    all_videos = all_videos[b:e]
    for video_path in all_videos:
        convert(video_path)


def check_converted():
    path = '/scratch/users/gabras/data/chalearn30/all_data/13kjwEtSyXc.003.h5'
    h5 = h5py.File(path, 'r')
    print(len(list(h5.keys())))
    print(len(h5['faces'][:]))
    h5.close()


def move_cluster_completeness():
    p_main = '/scratch/users/gabras/data/chalearn30/all_data'  # probably all complete
    p_cluster = '/scratch/users/gabras/data/chalearn30/chalearn30/all_data'
    main_list = os.listdir(p_main)
    cluster_list = os.listdir(p_cluster)
    diff = set(cluster_list) - set(main_list)

    cnt = 0
    completed_which_moved = 0

    # get all the mp4s
    all_train = DU.get_all_videos('train')

    for n in tqdm(diff):
        cnt += 1
        h5 = os.path.join(p_cluster, n)
        with h5py.File(h5, 'r') as mf:
            frames_h5 = len(mf.keys()) - 1

        n_base = n.split('.h5')[0]
        frames_mp4 = 0

        # find matching mp4
        for v in all_train:
            if n_base in v:
                frames_mp4 = DU.mp4_to_arr(v).shape[0]
                break

        if frames_mp4 == 0:
            print('ohboi')
            return

        if frames_h5 == frames_mp4:
            src = os.path.join(p_cluster, n)
            dst = os.path.join(p_main, n)
            # shutil.move(src=src, dst=dst)
            completed_which_moved += 1
            print(completed_which_moved, cnt)

    print('tot diff: %d')


# move_cluster_completeness()


def get_left_off_index():
    p_main = '/scratch/users/gabras/data/chalearn30/all_data'  # probably all complete
    p_cluster = '/scratch/users/gabras/data/chalearn30/chalearn30/all_data'
    main_list = os.listdir(p_main)
    cluster_list = os.listdir(p_cluster)
    diff = set(cluster_list) - set(main_list)

    # not_complete = set(cluster_list) - set(diff)

    all_train = DU.get_all_videos('train')

    indices_1000_2000 = []
    indices_2000_3000 = []
    indices_3000_4000 = []
    indices_4000_5000 = []
    indices_5000_6000 = []

    def do_thing(t):
        for i, v in enumerate(all_train):
            if t in v:
                # print(i)
                if i < 2000:
                    indices_1000_2000.append(i)
                elif i < 3000:
                    indices_2000_3000.append(i)
                elif i < 4000:
                    indices_3000_4000.append(i)
                elif i < 5000:
                    indices_4000_5000.append(i)
                elif i < 6000:
                    indices_5000_6000.append(i)
                break

    for p in diff:
        p_base = p.split('.h5')[0]
        do_thing(p_base)

    indices_1000_2000.sort()
    indices_2000_3000.sort()
    indices_3000_4000.sort()
    indices_4000_5000.sort()
    indices_5000_6000.sort()

    print(indices_1000_2000[0],indices_1000_2000[-1], '\n',
          indices_2000_3000[0],indices_2000_3000[-1], '\n',
          indices_3000_4000[0],indices_3000_4000[-1], '\n',
          indices_4000_5000[0],indices_4000_5000[-1], '\n',
          indices_5000_6000[0],indices_5000_6000[-1])


# get_left_off_index()
# 1000 1258
# 2000 2531
# 3000 3423
# 4000 4273
# 5000 5252

# parallel_convert('test', 0, 500, convert)  # done
# parallel_convert('test', 500, 1000, convert)  # done
# parallel_convert('test', 1000, 1500, convert)  # done

# parallel_convert('test', 1500, 2000, convert) # ? fails at 1924
# parallel_convert('test', 1924, 2000, convert, number_processes=10)  # done
# parallel_convert('train', 0, 500, convert) # ? fails at 425
# parallel_convert('train', 425, 500, convert, number_processes=10)  # done
# parallel_convert('train', 500, 1000, convert) # ? fails at 626
# parallel_convert('train', 626, 1000, convert, number_processes=20)  # done

# ---
# parallel_convert('val', 0, 500, convert, number_processes=20) # schmidhuber done
# parallel_convert('val', 500, 1000, convert, number_processes=20) # schmidhuber done
# parallel_convert('val', 1000, 1500, convert, number_processes=20) # schmidhuber done
# parallel_convert('val', 1500, 2000, convert, number_processes=20) # schmidhuber done


# issues with this since storage space ran out
# parallel_convert('train', 1000, 2000, convert, number_processes=20) # hinton busy, issues, partial
# parallel_convert('train', 2000, 3000, convert, number_processes=30) # turing busy, issues, partial
# parallel_convert('train', 3000, 4000, convert, number_processes=30) # archimedes busy, issues, partial
# parallel_convert('train', 4000, 5000, convert, number_processes=20) # ramachandran busy, issues, partial
# parallel_convert('train', 5000, 6000, convert, number_processes=20) # charcot busy, issues, partial
# fixing, all on schmidhuber
parallel_convert('train', 1258, 2000, convert, number_processes=15)
parallel_convert('train', 2531, 3000, convert, number_processes=15)
parallel_convert('train', 3423, 4000, convert, number_processes=15)
parallel_convert('train', 4273, 5000, convert, number_processes=15)
parallel_convert('train', 5252, 6000, convert, number_processes=15)
