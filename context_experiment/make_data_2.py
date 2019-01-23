import csv
import numpy as np
import os
import shutil
# import deepimpression2.chalearn10.mp4_to_h5 as MH
import deepimpression2.chalearn10.align_crop as AC
# from multiprocessing import Pool
import skvideo.io
import subprocess
import imageio
import time


def fix_some_videos():
    # _01AyUz9J9I.001.mp4
    # 3ySH8eJQRCI.002.mp4
    # AP0aklGHino.000.mp4
    # b-qfeGqPd04.001.mp4
    # erJNMcO69Eo.000.mp4
    # fDw_PAgW07o.004.mp4
    # j-9NUiMYbos.001.mp4
    # MK7KveAeb6Y.004.mp4
    # mpXOSY5dW7c.005.mp4
    # _Py_lytyY5A.005.mp4
    # S1AEj1kO5dc.000.mp4
    # sHVXhr7_EOs.001.mp4
    # TJD__22fOr0.000.mp4
    # ZGox7tevC6A.005.mp4
    pass




def get_audio():
    path_to_video = '/home/gabras/deployed/deepimpression2/context_experiment/faces/io0aNQE8En8.003.mp4'
    # path_to_video = '/home/gabras/deployed/deepimpression2/context_experiment/faces_normalized_fps/io0aNQE8En8.003.mp4'
    audio_name = 'io0aNQE8En8.003.mp3'
    # audio_name = 'io0aNQE8En8.003.wav'

    command = "ffmpeg -loglevel panic -ss 0 -t 5 -i %s -ab 160k -ac 2 -ar 44100 -vn -y -strict -2 %s" % (path_to_video, audio_name)
    # command = "ffmpeg -loglevel panic -ss 0 -t 5 -i %s -ab 160k -ac 2 -ar 44100 -vn -y -strict -2 %s" % (path_to_original, five_audio_name)
    subprocess.call(command, shell=True)

    # crop to 5 seconds
    # command = 'ffmpeg -ss 0 -t 30 -i %s %s' % (full_audio_name, five_audio_name)
    # subprocess.call(command, shell=True)

    # meta_data = skvideo.io.ffprobe(full_audio_name)
    # length = int(float(meta_data['audio']['@duration']))
    # print(length)

    meta_data = skvideo.io.ffprobe(audio_name)
    length = int(float(meta_data['audio']['@duration']))
    print(length)


get_audio()
