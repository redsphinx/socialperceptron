# ----------------------------------------------------------------
# created by https://gitlab.socsci.ru.nl/S.Iacob/3DDeepImpression/
# ----------------------------------------------------------------
import numpy as np
# import constants as pc
# import training_util
# import paths as pp
import skvideo.io
import subprocess
import h5py
import time
import librosa
import matplotlib.pyplot as plt
import os
import math
# from training_util import hdf5ToNp


FS = 16000.0


def mp4_to_arr(video_path):
    vid = skvideo.io.vread(video_path)
    # vid = skvideo.io.vread(video_path)
    vid = np.transpose(vid, [0, 3, 1, 2])
    vid = np.expand_dims(vid, 0)
    aud = librosa.load(video_path, FS)[0][None, None, None, :]
    return vid, aud


def mp4_to_meta(video_path):
    meta_data = skvideo.io.ffprobe(video_path)
    try:
        d = float(meta_data['video']['@duration'])
    except KeyError:
        print('KeyError on d')
        d = float(meta_data['video']['@duration'])
    try:
        fps = meta_data['video']['@r_frame_rate']
        fps = float(fps.split('/')[0])
    except KeyError:
        print('KeyError on fps')
        fps = meta_data['video']['@r_frame_rate']

    metadata = np.asarray([d, fps], dtype='float32')
    return metadata


def mp4_to_h5(video_path, destination_path):
    # read data
    vid, aud = mp4_to_arr(video_path)
    # read metadata
    metadata = mp4_to_meta(video_path)
    # saving data
    with h5py.File(destination_path, 'a') as my_file:
        my_file.create_dataset(name='video', data=vid)
        my_file.create_dataset(name='audio', data=aud)
        my_file.create_dataset(name='metadata', data=metadata)
        # read data
        # with h5py.File(destination_path, 'r') as my_file:
        #     vid = my_file['video'][:]
        #     aud = my_file['audio'][:]
        #     metadata = my_file['metadata'][:]
        #     print(vid.shape)
        #     print(aud.shape)
        #     print(metadata)


def extract_hdf5_from_dir(from_folder, to_folder):
    video_names = os.listdir(from_folder)
    for name in video_names:
        new_name = name.split('.mp4')[0]
        new_name += '.h5'
        destination_path = os.path.join(to_folder, new_name)
        video_path = os.path.join(from_folder, name)
        mp4_to_h5(video_path, destination_path)


# Copied from https://stackoverflow.com/questions/1392413/calculating-a-directorys-size-using-python
def du(path):
    """disk usage in human readable format (e.g. '2,1GB')"""
    return subprocess.check_output(['du', '-sh', path]).split()[0].decode('utf-8')


def preprocess_training_data():
    from_data = pp.training_data
    to_data = pp.hdf5_data
    dirs = os.listdir(from_data)
    for d in dirs:
        dir1 = os.path.join(from_data, d)
        dirs1 = os.listdir(dir1)
        to_dir = os.path.join(to_data, d)
        for dir2 in dirs1:
            from_path = os.path.join(dir1, dir2)
            to_path = os.path.join(to_dir, dir2)
            time_dir = time.time()
            # Actually extracting h5 from mp4
            extract_hdf5_from_dir(from_path, to_path)
            time_dir = time.time() - time_dir
            size = du(to_path)
            report = 'Directory %s took %f sec and %s' % (dir2, time_dir, size)
            print(report)


def preprocess_val_data():
    from_data = pp.validation_data
    to_data = pp.hdf5_val_data
    dirs = os.listdir(from_data)
    for d in dirs:
        val_dir = os.path.join(from_data, d)
        time_dir = time.time()
        size = du(to_data)
        extract_hdf5_from_dir(val_dir, to_data)
        time_dir = time.time() - time_dir
        report = 'Directory %s took %f sec' % (d, time_dir)
        print(report)


        # preprocess_val_data()
