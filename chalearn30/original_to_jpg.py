import time
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
import skvideo.io
import deepimpression2.chalearn20.constants as C2


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
# parallel_convert('train', 1000, 2000, convert, number_processes=20) # hinton busy, issues, partial


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
    # print(len(diff))

    cnt = 0
    completed_which_moved = 0

    # get all the mp4s
    all_train = DU.get_all_videos('train')

    for n in diff:
        cnt += 1
        print(cnt)
        h5 = os.path.join(p_cluster, n)
        try:
            mf = h5py.File(h5, 'r')
            # with h5py.File(h5, 'r') as mf:
            frames_h5 = len(mf.keys()) - 1
            mf.close()
            print('opened and closed')

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
                shutil.move(src=src, dst=dst)
                completed_which_moved += 1
                print(completed_which_moved, cnt)
            else:
                print('frames should be %d, but are %d' % (frames_mp4, frames_h5))
        except (OSError, RuntimeError) as e:
            print(h5, 'failed', e)


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

    # print(indices_1000_2000[0],indices_1000_2000[-1], '\n',
    #       indices_2000_3000[0],indices_2000_3000[-1], '\n',
    #       indices_3000_4000[0],indices_3000_4000[-1], '\n',
    #       indices_4000_5000[0],indices_4000_5000[-1], '\n',
    #       indices_5000_6000[0],indices_5000_6000[-1])

    all_train = np.array(all_train)
    all_train_1000_2000 = list(all_train[indices_1000_2000])
    all_train_2000_3000 = list(all_train[indices_2000_3000])
    all_train_3000_4000 = list(all_train[indices_3000_4000])
    all_train_4000_5000 = list(all_train[indices_4000_5000])
    all_train_5000_6000 = list(all_train[indices_5000_6000])

    # return ones that have already been converted
    return all_train_1000_2000, all_train_2000_3000, all_train_3000_4000, all_train_4000_5000, all_train_5000_6000


def get_missing():
    p_main = '/scratch/users/gabras/data/chalearn30/all_data'  # probably all complete
    p_cluster = '/scratch/users/gabras/data/chalearn30/chalearn30/all_data'
    main_list = os.listdir(p_main)
    cluster_list = os.listdir(p_cluster)
    diff = set(cluster_list) - set(main_list)

    all_train = DU.get_all_videos('train')
    missing = []

    def do_thing(t):
        for i, v in enumerate(all_train):
            if t in v:
                missing.append(v)

    for p in diff:
        p_base = p.split('.h5')[0]
        do_thing(p_base)

    return missing


def parallel_convert_mod(which, b, e, func, number_processes=20):
    all_videos = DU.get_all_videos(which)
    a12, a23, a34, a45, a56 = get_left_off_index()

    pool = Pool(processes=number_processes)
    all_videos = all_videos[b:e]
    if b == 1000:
        all_videos = list(set(all_videos) - set(a12))
    elif b == 2000:
        all_videos = list(set(all_videos) - set(a23))
    elif b == 3000:
        all_videos = list(set(all_videos) - set(a34))
    elif b == 4000:
        all_videos = list(set(all_videos) - set(a45))
    elif b == 5000:
        all_videos = list(set(all_videos) - set(a56))

    pool.apply_async(func)
    pool.map(func, all_videos)


def parallel_convert_missing(func, number_processes=30):
    all_videos = get_missing()
    pool = Pool(processes=number_processes)
    pool.apply_async(func)
    pool.map(func, all_videos)


def controleren(b, e):
    def get_frames_mp4(bn):
        for j, v in enumerate(all_videos):
            if bn in v:
                return int(skvideo.io.ffprobe(v)['video']['@nb_frames']), v

    def get_frames_h5(p):
        with h5py.File(p, 'r') as mf:
            frs = len(mf.keys()) - 1
        return frs

    still_todo = []

    all_videos = DU.get_all_videos('train') + DU.get_all_videos('test') + DU.get_all_videos('val')
    all_h5 = os.listdir(P.CHALEARN30_ALL_DATA)

    for i, h5 in enumerate(all_h5[b:e]):
        print(i)
        base_name = h5.split('.h5')[0]
        h5_path = os.path.join(P.CHALEARN30_ALL_DATA, h5)

        h5_frames = get_frames_h5(h5_path)
        mp4_frames, v_path = get_frames_mp4(base_name)

        if h5_frames != mp4_frames:
            still_todo.append(v_path)
            print('ohno, %s' % (v_path))

    print('still todo', len(still_todo))
    return still_todo


# b = controleren(1340, 2000)
# b = controleren(2000, 3000)
# b = controleren(3000, 4000)
# b = controleren(4000, 5000)
# b = controleren(5000, 6000)
# b = controleren(6000, 7000)
# b = controleren(7000, 8000)
# b = controleren(8000, 9000)
# b = controleren(9000, 10000)

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
# parallel_convert_mod('train', 1000, 2000, convert, number_processes=15)
# parallel_convert_mod('train', 2000, 3000, convert, number_processes=15)
# parallel_convert_mod('train', 3000, 4000, convert, number_processes=15)
# parallel_convert_mod('train', 4000, 5000, convert, number_processes=15)
# parallel_convert_mod('train', 5000, 6000, convert, number_processes=15)

# convert missing
# parallel_convert_missing(convert)

def fix_optface(optface):
    if optface[3] > C2.H:
        optface[3] = C2.H
    if optface[2] > C2.W:
        optface[2] = C2.W
    if optface[1] < 0:
        optface[1] = 0
    if optface[0] < 0:
        optface[0] = 0

    return optface


def convert_h5_to_jpg(b, e):
    print('b=%d, e=%d' % (b, e))

    src_face = P.CHALEARN_ALL_DATA_20_2
    src_else = P.CHALEARN30_ALL_DATA

    dest = [P.CHALEARN_JPG_FACE, P.CHALEARN_JPG_BG, P.CHALEARN_JPG_FULL_FRAME]

    for i in range(1,2):
        if i == 0:
            print('extracting face')
            src_files = os.listdir(src_face)
            src_files.sort()
            for j in tqdm(range(b, e)):
                v = src_files[j]
                d_path = os.path.join(dest[i], v.split('.h5')[0])
                if not os.path.exists(d_path):
                    os.mkdir(d_path)

                p_src = os.path.join(src_face, v)
                src_h5 = h5py.File(p_src, 'r')

                for frame in src_h5.keys():
                    image = src_h5[frame][:]
                    image = np.transpose(image, (1, 2, 0))
                    img = Image.fromarray(image, mode='RGB')
                    img = img.resize((C2.RESIDE, C2.RESIDE))
                    name = os.path.join(d_path, frame+'.jpg')
                    img.save(name)
        else:
            print('extracting full frame and bg')
            src_files = os.listdir(src_else)
            src_files.sort()
            for j in tqdm(range(b, e)):
                v = src_files[j]
                d_path_bg = os.path.join(dest[1], v.split('.h5')[0])
                d_path_ff = os.path.join(dest[2], v.split('.h5')[0])

                if not os.path.exists(d_path_bg):
                    os.mkdir(d_path_bg)
                if not os.path.exists(d_path_ff):
                    os.mkdir(d_path_ff)

                p_src = os.path.join(src_else, v)
                src_h5 = h5py.File(p_src, 'r')

                for frame in src_h5.keys():
                    if frame != 'faces':
                        image = src_h5[frame][:]
                        image = image[0]
                        try:
                            image = np.transpose(image, (1, 2, 0))
                        except ValueError:
                            print('hereh')
                        img = Image.fromarray(image, mode='RGB')

                        # save full frame
                        name_ff = os.path.join(d_path_ff, frame+'.jpg')
                        img.save(name_ff)

                        # take out face, crop bg
                        optface = src_h5['faces'][int(frame)]
                        optface = fix_optface(optface)

                        image = src_h5[frame][:]
                        px_mean = np.mean(image, 2)
                        px_mean = np.mean(px_mean, 2)

                        for ii in range(optface[0], optface[2]):
                            for jj in range(optface[1], optface[3]):
                                try:
                                    image[:, :, jj, ii] = px_mean
                                except IndexError:
                                    print(2, IndexError, jj, ii)

                        image = image.astype(np.uint8)

                        if optface[0] > (456 - optface[2]):
                            left = 0
                            right = 256
                        else:
                            left = 200
                            right = 456

                        image = np.transpose(image[0], (1, 2, 0))
                        img = Image.fromarray(image, mode='RGB')
                        img = img.crop((left, 0, right, 256))  # left, upper, right, and lower

                        # save bg
                        name_bg = os.path.join(d_path_bg, frame + '.jpg')
                        img.save(name_bg)


# first 1000
s = time.time()
# convert_h5_to_jpg(0, 100)
convert_h5_to_jpg(100, 200)
# convert_h5_to_jpg(200, 300)
# convert_h5_to_jpg(300, 400)
# convert_h5_to_jpg(400, 500)
# convert_h5_to_jpg(500, 600)
# convert_h5_to_jpg(600, 700)
# convert_h5_to_jpg(700, 800)
# convert_h5_to_jpg(800, 900)
# convert_h5_to_jpg(900, 1000)
print('%f min' % ((time.time()-s)/60))
