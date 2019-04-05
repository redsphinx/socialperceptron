import numpy as np
import h5py as h5
import os
from tqdm import tqdm
from PIL import Image
import time

import chainer
from chainer.backends.cuda import to_gpu, to_cpu

from deepimpression2.model_59 import Deepimpression
from deepimpression2.model_59 import LastLayers
import deepimpression2.paths as P
import deepimpression2.chalearn20.constants as C2


def load_models(model_type, gpu, dev):
    all_models = None

    if model_type == 'deepimpression':
        model_names = [['epoch_39_59_O', 'epoch_19_59_C', 'epoch_99_59_E', 'epoch_89_59_A', 'epoch_19_59_S'],  # face
                       ['epoch_89_60_O', 'epoch_79_60_C', 'epoch_99_60_E', 'epoch_89_60_A', 'epoch_89_60_S'],  # bg
                       ['epoch_99_105_O', 'epoch_99_105_C', 'epoch_19_105_E', 'epoch_99_61_A', 'epoch_9_61_S']  # all
                       ]

        all_models = []
        for i, names in enumerate(model_names):
            tmp = []
            if i < 2:
                for j, name in enumerate(names):
                    my_model = Deepimpression()
                    p = os.path.join(P.MODELS, name)
                    chainer.serializers.load_npz(p, my_model)
                    if gpu:
                        my_model = my_model.to_gpu(device=dev)
                    tmp.append(my_model)
                all_models.append(tmp)
            elif i == 2:
                for j, name in enumerate(names):
                    my_model = LastLayers()
                    p = os.path.join(P.MODELS, name)
                    chainer.serializers.load_npz(p, my_model)
                    if gpu:
                        my_model = my_model.to_gpu(device=dev)
                    tmp.append(my_model)
                all_models.append(tmp)

    return all_models


def get_frame(key, frame):
    # face
    face_path = os.path.join(P.CHALEARN_ALL_DATA_20_2, '%s.h5' % key)
    f = h5.File(face_path, 'r')
    face = f[str(frame)][:]  # shape=(c, h, w)
    f.close()

    # bg
    bg_path = os.path.join(P.CHALEARN_TEST_SCHMID, '%s.h5' % key)
    b = h5.File(bg_path, 'r')
    bg = b[str(frame)][:]  # shape=(1, c, h, w)
    optface = b['faces'][frame]
    b.close()

    return face, bg, optface


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


def resize_face(image):
    image = np.transpose(image, (1, 2, 0))
    img = Image.fromarray(image, mode='RGB')
    img = img.resize((C2.RESIDE, C2.RESIDE))
    img = np.array(img)
    image = np.transpose(img, (2, 0, 1))
    image = np.expand_dims(image, 0)
    return image


def resize_bg(image, optface):
    px_mean = np.mean(image, 2)
    px_mean = np.mean(px_mean, 2)

    for i in range(optface[0], optface[2]):
        for j in range(optface[1], optface[3]):
            try:
                image[:, :, j, i] = px_mean
            except IndexError:
                print(2, IndexError, j, i)

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
    img = np.array(img)
    image = np.transpose(img, (2, 0, 1))
    image = np.expand_dims(image, 0)

    return image


def run(all_models, id_frames, labels, dev, batch_size, gpu):
    all_traits = ['O', 'C', 'E', 'A', 'S']

    for k in labels.keys():

        BOSS_ARRAY = np.zeros(shape=(3, 5))  # holds mean prediction for single video

        k = k.split('.mp4')[0]
        if k != '4ZlcaXadwlo.005':  # video with no frames

            print('video: %s' % k)

            num_frames = id_frames[k][0] - 1

            steps = num_frames // batch_size

            worker_array = np.zeros(shape=(3, 5, num_frames-(num_frames%batch_size)))

            for s in tqdm(range(steps)):
                data = np.zeros(shape=(batch_size, 2, 3, 256, 256), dtype=np.float32)

                start = s * batch_size
                end = start + batch_size

                cnt = 0
                for f in range(start, end):
                    face, bg, optface = get_frame(k, f)
                    optface = fix_optface(optface)

                    face = resize_face(face)
                    bg = resize_bg(bg, optface)

                    data[cnt] = face, bg
                    cnt += 1

                if gpu:
                    data = to_gpu(data, device=dev)

                # for trait in traits
                for i, t in enumerate(all_traits):
                    face_model = all_models[0][i]
                    bg_model = all_models[1][i]
                    my_model = all_models[2][i]

                    with chainer.using_config('train', False):
                        face_pred, face_features = face_model(data[:, 0])
                        bg_pred, bg_features = bg_model(data[:, 1])
                        all_pred = my_model(bg_features, face_features)

                    worker_array[0][i][start:end] = to_cpu(face_pred.data).flatten()
                    worker_array[1][i][start:end] = to_cpu(bg_pred.data).flatten()
                    worker_array[2][i][start:end] = to_cpu(all_pred.data).flatten()

            BOSS_ARRAY[0] = np.mean(worker_array[0], axis=-1)
            BOSS_ARRAY[1] = np.mean(worker_array[1], axis=-1)
            BOSS_ARRAY[2] = np.mean(worker_array[2], axis=-1)

            file_names_face = ['pred_132_O', 'pred_132_C', 'pred_132_E', 'pred_132_A', 'pred_132_S']
            file_names_bg = ['pred_133_O', 'pred_133_C', 'pred_133_E', 'pred_133_A', 'pred_133_S']
            file_names_all = ['pred_134_O', 'pred_134_C', 'pred_134_E', 'pred_134_A', 'pred_134_S']

            # all_files = [file_names_face, file_names_bg, file_names_all]
            # for i, mode in enumerate(all_files):
            #     for j, f in enumerate(mode):
            #         f = f + '.txt'
            #         path = os.path.join(P.LOG_BASE, f)
            #         line = '%f\n' % BOSS_ARRAY[i][j]
            #         with open(path, 'a') as my_file:
            #             my_file.write(line)


def main_loop():
    start = time.time()
    print('s\n t\n  a\n   r\n    t')
    model_type = 'deepimpression'
    use_gpu = True
    device = 1
    batch = 32

    all_models = [] #load_models(model_type, use_gpu, device)
    labels = h5.File(P.CHALEARN_TEST_LABELS_20, 'r')
    id_frames = h5.File(P.NUM_FRAMES, 'r')

    run(all_models, id_frames, labels, dev=device, batch_size=batch, gpu=use_gpu)

    print('total time: %d minutes' % ((time.time()-start)/60))
    print('d\n o\n  n\n   e')


main_loop()
