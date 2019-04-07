import numpy as np
import h5py as h5
import os
from tqdm import tqdm
import time

import chainer
from chainer.backends.cuda import to_gpu, to_cpu

import torch
import torchvision.transforms as transforms

from deepimpression2.model_16 import Deepimpression
from deepimpression2.model_resnet18 import hackyResNet18
import deepimpression2.paths as P


def load_models(gpu, dev):
    all_models = []
    
    model_names = ['epoch_54_149', 'epoch_9_150', 'epoch_14_151']

    for i, name in enumerate(model_names):
        if i == 0:
            my_model = Deepimpression()
            p = os.path.join(P.MODELS, name)
            chainer.serializers.load_npz(p, my_model)
            if gpu:
                my_model = my_model.to_gpu(device=dev)
        else:
            if i == 1:
                my_model = hackyResNet18(5, True)
            else:
                my_model = hackyResNet18(5, False)
            p = os.path.join(P.MODELS, name)
            my_model.load_state_dict(torch.load(p))
            if gpu:
                my_model = my_model.cuda(dev)

        all_models.append(my_model)

    return all_models


def get_frame(key, frame):
    ff_path = os.path.join(P.CHALEARN_TEST_SCHMID, '%s.h5' % key)
    f = h5.File(ff_path, 'r')
    full_frame = f[str(frame)][:]  # shape=(1, c, h, w)
    f.close()

    return full_frame


def do_the_normalize(im):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    totensor = transforms.ToTensor()

    im = im[0]
    im = im.transpose((1, 2, 0))
    im = totensor(im)
    im = normalize(im)
    im = im.numpy()
    im = np.expand_dims(im, 0)
    return im


def arr_to_str(arr):
    assert arr.ndim == 1
    l = ''
    for i in arr:
        l += str(i) + ','

    l = l[0:-1]
    return l


def run(all_models, id_frames, labels, dev, batch_size):

    for k in labels.keys():
        BOSS_ARRAY = np.zeros(shape=(3, 5))  # holds mean prediction for single video

        k = k.split('.mp4')[0]
        if k != '4ZlcaXadwlo.005':  # video with no frames

            print('video: %s' % k)

            num_frames = id_frames[k][0] - 1

            steps = num_frames // batch_size

            worker_array = np.zeros(shape=(3, 5, num_frames - (num_frames % batch_size)))

            for s in tqdm(range(steps)):
                data_np = np.zeros(shape=(batch_size, 3, 256, 456), dtype=np.float32)
                data_pt = np.zeros(shape=(batch_size, 3, 256, 456), dtype=np.float32)

                start = s * batch_size
                end = start + batch_size

                cnt = 0
                for f in range(start, end):
                    full_frame = get_frame(k, f)
                    data_np[cnt] = full_frame
                    data_pt[cnt] = do_the_normalize(full_frame)

                    cnt += 1

                data_di = to_gpu(data_np, device=dev)
                data_rn_pt = torch.from_numpy(data_pt)
                data_rn_pt = data_rn_pt.cuda(dev)
                data_rn_np = torch.from_numpy(data_np)
                data_rn_np = data_rn_np.cuda(dev)

                di_model = all_models[0]
                rn_pt_model = all_models[1]
                rn_np_model = all_models[2]

                with chainer.using_config('train', False):
                    di_pred = di_model(data_di)
                
                rn_pt_model.eval()
                rn_np_model.eval()

                with torch.no_grad():
                    rn_pt_pred = rn_pt_model(data_rn_pt)
                    rn_np_pred = rn_np_model(data_rn_np)

                di_pred = to_cpu(di_pred.data)
                rn_pt_pred = np.array(rn_pt_pred.cpu().data)
                rn_np_pred = np.array(rn_np_pred.cpu().data)

                for i in range(5):
                    worker_array[0][i][start:end] = di_pred[:, i].flatten()
                    worker_array[1][i][start:end] = rn_pt_pred[:, i].flatten()
                    worker_array[2][i][start:end] = rn_np_pred[:, i].flatten()

            BOSS_ARRAY[0] = np.mean(worker_array[0], axis=-1)
            BOSS_ARRAY[1] = np.mean(worker_array[1], axis=-1)
            BOSS_ARRAY[2] = np.mean(worker_array[2], axis=-1)

            file_names_di = 'pred_172'
            file_names_rn_pt = 'pred_173'
            file_names_rn_np = 'pred_174'

            all_files = [file_names_di, file_names_rn_pt, file_names_rn_np]

            for i, f in enumerate(all_files):
                f = f + '.txt'
                path = os.path.join(P.LOG_BASE, f)
                line = arr_to_str(BOSS_ARRAY[i]) + '\n'
                with open(path, 'a') as my_file:
                    my_file.write(line)


def main_loop():
    start = time.time()
    print('s\n t\n  a\n   r\n    t')

    use_gpu = True
    device = 1
    batch = 32

    all_models = load_models(use_gpu, device)
    labels = h5.File(P.CHALEARN_TEST_LABELS_20, 'r')
    id_frames = h5.File(P.NUM_FRAMES, 'r')

    run(all_models, id_frames, labels, device, batch)

    print('total time: %d minutes' % ((time.time() - start) / 60))
    print('d\n o\n  n\n   e')


main_loop()
