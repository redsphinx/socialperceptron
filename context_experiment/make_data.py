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
from PIL import Image


path_to_videos_train = '/scratch/users/gabras/data/chalearn10/original_train'
path_to_videos_test = '/scratch/users/gabras/data/chalearn10/original_test'
path_to_videos_val = '/scratch/users/gabras/data/chalearn10/original_val'
csv_pairs = '/home/gabras/deployed/deepimpression2/context_experiment/pairs.csv'
all_video_path = '/home/gabras/deployed/deepimpression2/context_experiment/all_video_paths.txt'

path_to_original_videos = '/home/gabras/deployed/deepimpression2/context_experiment/videos'
path_to_faces = '/home/gabras/deployed/deepimpression2/context_experiment/faces'

path_to_completed = '/home/gabras/deployed/deepimpression2/context_experiment/completed'
path_to_normalized_videos = '/home/gabras/deployed/deepimpression2/context_experiment/normalized_videos'
path_to_cut_videos = '/home/gabras/deployed/deepimpression2/context_experiment/cut_videos'
path_to_faces_normalized_fps = '/home/gabras/deployed/deepimpression2/context_experiment/faces_normalized_fps'


def make_list_video_paths(path_to_videos):
    dir_videos = os.listdir(path_to_videos)
    for i in dir_videos:
        full_path = os.path.join(path_to_videos, i)
        if os.path.isdir(full_path):
            sub_dir_videos = os.listdir(full_path)
            for j in sub_dir_videos:
                sub_dir_videos_2 = os.path.join(full_path, j)
                videos = os.listdir(sub_dir_videos_2)
                for k in videos:
                    video_path = os.path.join(sub_dir_videos_2, k)
                    if k.split('.')[-1] == 'mp4':
                        with open(all_video_path, 'a') as my_file:
                            line = '%s,%s\n' % (video_path, k)
                            my_file.write(line)


def load_file(path, header=False):
    if os.path.exists(path):
        if header:
            list_video_paths = np.genfromtxt(path, dtype=str, delimiter=',', skip_header=0)
        else:
            list_video_paths = np.genfromtxt(path, dtype=str, delimiter=',', skip_header=1)
        return list_video_paths
    else:
        print('no such path')


def copy_correct_videos():
    if not os.path.exists(all_video_path):
        make_list_video_paths(path_to_videos_train)
        make_list_video_paths(path_to_videos_test)
        make_list_video_paths(path_to_videos_val)
    csv_pairs_uniques = load_file(csv_pairs, header=False)[:, 4:6]
    csv_pairs_uniques = list(set(np.ndarray.flatten(csv_pairs_uniques).tolist()))
    video_paths = load_file(all_video_path, header=True)
    fulls = video_paths[:, 0]
    names = list(video_paths[:, 1])
    cnt = 0
    for i in csv_pairs_uniques:
        try:
            full = fulls[names.index(i)]
            dest = os.path.join('/home/gabras/deployed/deepimpression2/context_experiment/videos', i)
            shutil.copy(full, dest)
        except ValueError:
            cnt += 1
            print(cnt)


def videos_to_faces(b, e):
    original_videos = os.listdir(path_to_normalized_videos)
    print('total: ', len(original_videos))
    print('b', b, 'e', e)
    for i in original_videos[b:e]:
        src = os.path.join(path_to_normalized_videos, i)
        fps = 30
        duration = 5
        dst = path_to_faces_normalized_fps
        AC.align_faces_in_video(data_path=src, save_location=dst, side=196, audio=True, frames=duration*fps)


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


def get_pairs():
    pair_csv = np.genfromtxt(csv_pairs, dtype=str, delimiter=',', skip_header=True)
    pair_names = pair_csv[:, -1]
    pair_names = [i.split('/')[-1] for i in pair_names]
    video_tuples = pair_csv[:, 4:6]
    return video_tuples, pair_names


def make_green_outline(frame):
    copy_frame = frame
    green = np.array([0, 255, 0])
    width = 3
    frame_shape = copy_frame.shape
    copy_frame[0:width, 0:frame_shape[0]] = green  # top
    copy_frame[frame_shape[1]-width:frame_shape[1], 0:frame_shape[0]] = green # bot
    copy_frame[0:frame_shape[1], 0:width] = green # left
    copy_frame[0:frame_shape[1], frame_shape[0]-width:frame_shape[0]] = green # right
    # frame_image = Image.fromarray(np.array(frame, dtype='uint8'), mode='RGB')
    # frame_image.save('testing_frame.jpg')
    return copy_frame


def get_audio(path):
    # create silent second
    path_silent = os.path.join(path_to_completed, 'silent.wav')
    command = "ffmpeg -f lavfi -i anullsrc=channel_layout=5.1:sample_rate=48000 -t 1 %s" % path_silent
    subprocess.call(command, shell=True)

    data_path = os.path.join(path_to_faces, path)
    name_video = path.split('.mp4')[0]
    name_audio = os.path.join(path_to_completed, '%s.wav' % name_video)
    # command = "ffmpeg -loglevel panic -i %s -ab 160k -ac 2 -ar 44100 -vn -y %s" % (data_path, name_audio)
    command = "ffmpeg -loglevel panic -ss 0 -t 5 -i %s -ab 160k -ac 2 -ar 44100 -vn -y -strict -2 %s" % (data_path, name_audio)
    subprocess.call(command, shell=True)

    # check if lengths are correct
    # for created wav, get metadata
    meta_data = skvideo.io.ffprobe(name_audio)
    length = int(float(meta_data['audio']['@duration']))
    if length != 5:
        # AC.remove_file(name_audio)
        if length == 4:
            # add 1 sec of silence at end
            new_name = os.path.join(path_to_completed, '%s_2.wav' % name_video)
            command = "ffmpeg -i %s -i %s -filter_complex '[0:0][1:0] concat=n=2:v=0:a=1[out]' -map '[out]' %s" % \
                      (name_audio, path_silent, new_name)
            # command = 'ffmpeg -i concat:"%s|%s" -codec copy %s' % (name_audio, name_audio, new_name)
            subprocess.call(command, shell=True)
            # remove original wav
            AC.remove_file(name_audio)
            # rename silenced one
            os.rename(new_name, name_audio)
        else:
            print('length = %d' % length)

    AC.remove_file(path_silent)
    print('audio extracted and saved')
    return name_audio


def merge_audio(left, right, name):
    command = "ffmpeg -i %s -i %s -filter_complex '[0:0][1:0] concat=n=2:v=0:a=1[out]' -map '[out]' %s" % \
              (left, right, name)
    subprocess.call(command, shell=True)


def glue_together():
    pairs, pair_name = get_pairs() # list of lists or tuples of strings
    h, w, c = 404, 1440, 3
    background = np.zeros((h, w, c)) # black
    image_size = 208
    left_image_y = 404 // 2 - 208 // 2
    left_image_x = 1440 // 4 - 208 // 2
    right_image_y = 404 // 2 - 208 // 2
    right_image_x = left_image_x + 1440 // 2

    # 5 seconds each, repeat each 2 times
    # get the videoframerates ?
    fps = 30
    frames = 10*fps

    for i, p in enumerate(pairs):
        if i < 1000:
            merged_video = np.zeros((frames, h, w, c), dtype='uint8')
            # vid_left = skvideo.io.vread(os.path.join(path_to_faces, p[0]))
            # vid_right = skvideo.io.vread(os.path.join(path_to_faces, p[1]))
            vid_left = skvideo.io.vread(os.path.join(path_to_faces_normalized_fps, p[0]))
            vid_right = skvideo.io.vread(os.path.join(path_to_faces_normalized_fps, p[1]))


            # if frames < half, outline left frame green, else outline right frame
            for f in range(frames // 2):
                base = background
                left_frame = make_green_outline(vid_left[f])
                base[left_image_y:left_image_y+image_size, left_image_x:left_image_x+image_size] = left_frame
                base[right_image_y:right_image_y+image_size, right_image_x:right_image_x+image_size] = vid_right[f]
                base = np.array(base, dtype='uint8')
                # save image
                # base_image = Image.fromarray(base, mode='RGB')
                # base_image.save('testing_side_by_side.jpg')
                merged_video[f] = base

            vid_left = skvideo.io.vread(os.path.join(path_to_faces_normalized_fps, p[0]))
            vid_right = skvideo.io.vread(os.path.join(path_to_faces_normalized_fps, p[1]))

            for f in range(frames // 2):
                base = background
                right_frame = make_green_outline(vid_right[f])
                base[left_image_y:left_image_y+image_size, left_image_x:left_image_x+image_size] = vid_left[f]
                base[right_image_y:right_image_y+image_size, right_image_x:right_image_x+image_size] = right_frame
                base = np.array(base, dtype='uint8')
                merged_video[f + (frames // 2)] = base

            # create video without sound
            path_tmp = os.path.join(path_to_completed, pair_name[i])
            imageio.mimwrite(path_tmp, merged_video, fps=fps)

            # get the audio
            audio_left = get_audio(p[0])
            audio_right = get_audio(p[1])
            name_audio = os.path.join(path_to_completed, '%s.wav' % pair_name[i].split('.')[0])
            merge_audio(audio_left, audio_right, name_audio)

            # put audio and video together + save
            time.sleep(1)
            avi_vid_name = os.path.join(path_to_completed, '%s.avi' % pair_name[i].split('.')[0])
            vid_name = os.path.join(path_to_completed, pair_name[i])

            AC.add_audio(vid_name, name_audio, avi_vid_name)
            # command = "ffmpeg -loglevel panic -i %s -i %s -codec copy -shortest -y %s" % (vid_name, name_audio,
            #                                                                               avi_vid_name)
            # subprocess.call(command, shell=True)
            # remove first mp4
            AC.remove_file(vid_name)
            # convert avi to mp4
            AC.avi_to_mp4(avi_vid_name, vid_name)
            # remove the wav file
            AC.remove_file(name_audio)
            AC.remove_file(audio_left)
            AC.remove_file(audio_right)
            # remove the avi file
            AC.remove_file(avi_vid_name)
            print('done with %s' % pair_name[i])


def normalize_videos():
    videos = os.listdir(path_to_original_videos)
    for i, v in enumerate(videos):
        src = os.path.join(path_to_original_videos, v)
        dst = os.path.join(path_to_normalized_videos, v)
        frame_rate = 30
        command = 'ffmpeg -y -i %s -strict -2 -r %d %s ' % (src, frame_rate, dst)
        subprocess.call(command, shell=True)


def cut_videos():
    # TODO: doesn't work
    videos = os.listdir(path_to_normalized_videos)
    for i, v in enumerate(videos):
        if i < 3:
            src = os.path.join(path_to_normalized_videos, v)
            dst = os.path.join(path_to_cut_videos, v)
            command = 'ffmpeg -i %s -ss 00:00:01 -t 00:05:01 -async 1 -strict -2 %s' % (src, dst)
            subprocess.call(command, shell=True)


if __name__ == '__main__':
    # b = 180
    # e = 200
    # print(b, e)
    # videos_to_faces(b, e)

    # normalize_videos()
    # videos_to_faces(b, e)
    glue_together()

