import numpy as np
from skimage import transform
from skimage import img_as_ubyte
import paths
import os
import dlib
import cv2
import skvideo.io
import imageio
import time
import subprocess
from multiprocessing import Pool
from deepimpression2 import paths as P


def add_audio(vid_name, name_audio, avi_vid_name):
    command = "ffmpeg -loglevel panic -i %s -i %s -codec copy -shortest -y %s" % (vid_name, name_audio,
                                                                                  avi_vid_name)
    subprocess.call(command, shell=True)


def avi_to_mp4(old_path, new_path):
    command = "ffmpeg -loglevel panic -i %s -strict -2 %s" % (old_path, new_path)
    subprocess.call(command, shell=True)


def remove_file(file_path):
    forbidden = ['/', '/home', '/home/gabi', '*', '']
    if file_path in forbidden:
        print('ERROR: removing this file will lead to catastrophic error')
    else:
        command = "mv %s /tmp" % file_path
        subprocess.call(command, shell=True)


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def get_template_landmark():
    file_path = paths.TEMPLATE
    template = list(np.genfromtxt(file_path, dtype=str))
    num_landmarks = len(template)
    template_arr = np.zeros((num_landmarks, 2), dtype='int')
    for i in range(num_landmarks):
        x, y = template[i].strip().split(',')
        template_arr[i] = [int(x), int(y)]
    return template_arr


class FaceAligner:
    def __init__(self, predictor, desiredFaceWidth, desiredLeftEye=(0.35, 0.35), desiredFaceHeight=None):
        self.predictor = predictor
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight

        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth

    def align_to_template_similarity(self, image, gray, rect):
        template_landmarks = get_template_landmark()
        detected_landmarks = shape_to_np(self.predictor(gray, rect))

        tf = transform.estimate_transform('similarity', detected_landmarks, template_landmarks)
        result = img_as_ubyte(transform.warp(image, inverse_map=tf.inverse, output_shape=(self.desiredFaceWidth,
                                                                                          self.desiredFaceWidth, 3)))
        return result


def find_largest_face(face_rectangles):
    number_rectangles = len(face_rectangles)

    if number_rectangles == 0:
        return None
    elif number_rectangles == 1:
        return face_rectangles[0]
    else:
        largest = 0
        which_rectangle = None
        for i in range(number_rectangles):
            r = face_rectangles[i]
            # it's a square so only one side needs to be checked
            width = r.right() - r.left()
            if width > largest:
                largest = width
                which_rectangle = i
        # print('rectangle %d is largest with a side of %d' % (which_rectangle, largest))
        return face_rectangles[which_rectangle]


def find_face_simple(image):
    detector = dlib.get_frontal_face_detector()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_rectangles = detector(gray, 2)
    if len(face_rectangles) == 0:
        print('no face detected in the generated image')
        return [0, 0, 0, 0]
        # return xp.zeros((image.shape), dtype=xp.uint8)
    largest_face_rectangle = find_largest_face(face_rectangles)

    return [largest_face_rectangle.left(), largest_face_rectangle.top(),
            largest_face_rectangle.right(), largest_face_rectangle.bottom()]


def align_face(image, xp):
    predictor = paths.PREDICTOR
    predictor = dlib.shape_predictor(predictor)
    detector = dlib.get_frontal_face_detector()
    fa = FaceAligner(predictor, desiredFaceWidth=196)  # 208?
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_rectangles = detector(gray, 2)
    if len(face_rectangles) == 0:
        print('no face detected in the generated image')
        return None
        # return xp.zeros((image.shape), dtype=xp.uint8)
    largest_face_rectangle = find_largest_face(face_rectangles)
    face_aligned = fa.align_to_template_similarity(image, gray, largest_face_rectangle)
    return face_aligned.astype(xp.uint8)


def align_faces_in_video(data_path, frames=None, audio=True, side=196):

    # normal case use this
    # save_location = P.CHALEARN_FACES_TEST_TIGHT

    # only for fixing missing 89
    save_location = '/scratch/users/gabras/data/chalearn10/server_1911'

    if os.path.exists(data_path):
        video_capture = skvideo.io.vread(data_path)
        meta_data = skvideo.io.ffprobe(data_path)
        fps = str(meta_data['video']['@avg_frame_rate'])
        fps = int(fps.split('/')[0][:2])
        # print('fps: %s' % fps)

        name_video = data_path.split('/')[-1].split('.mp4')[0]
        # name_video = data_path.split('/')[-1].split('.MTS')[0]
        name_audio = None

        video_capture = np.array(video_capture, dtype=np.uint8)
        if audio:
            name_audio = os.path.join(save_location, '%s.wav' % name_video)
            command = "ffmpeg -loglevel panic -i %s -ab 160k -ac 2 -ar 44100 -vn -y %s" % (data_path, name_audio)
            subprocess.call(command, shell=True)

        if frames is None:
            frames = np.shape(video_capture)[0]

        channels = 3

        # new_video_array = np.zeros((20, side, side, channels), dtype='uint8')
        new_video_array = np.zeros((frames, side, side, channels), dtype='uint8')

        # for i in range(20):
        for i in range(frames):
            if i % 20 == 0:
                print('%s: %s of %s' % (name_video, i, frames))
            frame = video_capture[i]
            new_frame = align_face(frame, xp=np)

            # if no face detected, copy face from previous frame
            if new_frame is None:
                if (i - 1) == -1:
                    new_frame = np.zeros((side, side, channels))
                else:
                    new_frame = new_video_array[i - 1]

            new_frame = np.array(new_frame, dtype='uint8')
            new_video_array[i] = new_frame

        print('END %s' % name_video)
        # vid_name = os.path.join(save_location, '%s.mp4' % name_video)
        # comment for testing
        vid_name = os.path.join(save_location, '%s.mp4' % name_video)
        imageio.mimwrite(vid_name, new_video_array, fps=fps)
        if audio:
            # add audio to the video
            time.sleep(1)
            avi_vid_name = os.path.join(save_location, '%s.avi' % name_video)
            add_audio(vid_name, name_audio, avi_vid_name)
            command = "ffmpeg -loglevel panic -i %s -i %s -codec copy -shortest -y %s" % (vid_name, name_audio,
                                                                                          avi_vid_name)
            subprocess.call(command, shell=True)
            # remove first mp4
            remove_file(vid_name)
            # convert avi to mp4
            avi_to_mp4(avi_vid_name, vid_name)
            # remove the wav file
            remove_file(name_audio)
            # remove the avi file
            remove_file(avi_vid_name)
    else:
        print('Error: data_path does not exist')


def parallel_align(b, e, func, number_processes=10):
    # usage in mp4_to_h5.py: AC.parallel_align(0, 200, AC.align_faces_in_video)
    print(b, e)
    all_test_paths = []
    f1 = os.listdir(P.CHALEARN_TEST_ORIGINAL)
    for i in f1:
        f1_path = os.path.join(P.CHALEARN_TEST_ORIGINAL, i)
        f2 = os.listdir(f1_path)
        for j in f2:
            f2_path = os.path.join(f1_path, j)
            videos = os.listdir(f2_path)
            for v in videos:
                video_path = os.path.join(f2_path, v)
                all_test_paths.append(video_path)
    # func has to be align_faces_in_video
    pool = Pool(processes=number_processes)
    list_path_all_videos = all_test_paths[b:e]
    # make folder in sa
    pool.apply_async(func)
    pool.map(func, list_path_all_videos)


def fix_missing_89_parallel(func, number_processes=10):
    # fix_missing_89_parallel(align_faces_in_video, number_processes=20)
    all_test_paths = []
    f1 = os.listdir(P.CHALEARN_TEST_ORIGINAL)
    for i in f1:
        f1_path = os.path.join(P.CHALEARN_TEST_ORIGINAL, i)
        f2 = os.listdir(f1_path)
        for j in f2:
            f2_path = os.path.join(f1_path, j)
            videos = os.listdir(f2_path)
            for v in videos:
                video_path = os.path.join(f2_path, v)
                all_test_paths.append(video_path)

    all_test_names = [i.split('/')[-1] for i in all_test_paths]

    loc = '/scratch/users/gabras/data/chalearn10/server_1911'
    processed = os.listdir(loc)

    todo = list(set(all_test_names) - set(processed))

    todo_location = []

    for i in todo:
        for j in all_test_paths:
            if j.split('/')[-1] == i:
                todo_location.append(j)

    pool = Pool(processes=number_processes)
    pool.apply_async(func)
    pool.map(func, todo_location)

