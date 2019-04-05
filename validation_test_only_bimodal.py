import chainer
import numpy as np
from deepimpression2.model_59 import Deepimpression
from deepimpression2.model_59 import LastLayers
import deepimpression2.constants as C
from chainer.functions import sigmoid_cross_entropy, mean_absolute_error, softmax_cross_entropy
from chainer.optimizers import Adam
import h5py as h5
import deepimpression2.paths as P
# import deepimpression2.chalearn20.data_utils as D
import deepimpression2.chalearn30.data_utils as D
import time
from chainer.backends.cuda import to_gpu, to_cpu
import deepimpression2.util as U
import os
import cupy as cp
from chainer.functions import expand_dims
from random import shuffle
from tqdm import tqdm


def initialize(which, model_name, bg, face):
    my_model = LastLayers()
    load_model = True
    if load_model:
        p = os.path.join(P.MODELS, model_name)
        chainer.serializers.load_npz(p, my_model)
        print('%s loaded' % model_name)

    bg_model = Deepimpression()
    p = os.path.join(P.MODELS, bg)
    chainer.serializers.load_npz(p, bg_model)
    print('bg model %s loaded' % bg)

    face_model = Deepimpression()
    p = os.path.join(P.MODELS, face)
    chainer.serializers.load_npz(p, face_model)
    print('face model %s loaded' % face)

    my_optimizer = Adam(alpha=0.0002, beta1=0.5, beta2=0.999, eps=10e-8)
    my_optimizer.setup(my_model)

    if C.ON_GPU:
        my_model = my_model.to_gpu(device=C.DEVICE)
        bg_model = bg_model.to_gpu(device=C.DEVICE)
        face_model = face_model.to_gpu(device=C.DEVICE)

    print('Initializing')
    print('model initialized with %d parameters' % my_model.count_params())

    epochs = 1

    if which == 'val':
        labels = h5.File(P.CHALEARN_VAL_LABELS_20, 'r')
        steps = len(labels) // C.VAL_BATCH_SIZE
    elif which == 'test':
        labels = h5.File(P.CHALEARN_TEST_LABELS_20, 'r')
        steps = len(labels) // C.TEST_BATCH_SIZE
    else:
        print('which is not correct')
        labels = None
        steps = None

    loss = []
    pred_diff = np.zeros((1, 1), float)

    id_frames = h5.File(P.NUM_FRAMES, 'r')

    return my_model, bg_model, face_model, my_optimizer, epochs, labels, steps, loss, pred_diff, id_frames


def run(which, steps, which_labels, frames, model, bg_model, face_model, optimizer, pred_diff, loss_saving, trait,
        ordered, save_all_results, twostream, same_frame, record_loss, record_predictions):
    print('steps: ', steps)
    assert(which in ['train', 'test', 'val'])
    assert(trait in ['O', 'C', 'E', 'A', 'S'])

    if which == 'train':
        which_batch_size = C.TRAIN_BATCH_SIZE
    elif which == 'val':
        which_batch_size = C.VAL_BATCH_SIZE
    elif which == 'test':
        which_batch_size = C.TEST_BATCH_SIZE

    loss_tmp = []
    pd_tmp = np.zeros((steps, 1), dtype=float)
    _labs = list(which_labels)

    preds = np.zeros((steps, 1), dtype=float)

    if not ordered:
        shuffle(_labs)

    ts = time.time()
    for s in tqdm(range(steps)):
        labels_selected = _labs[s * which_batch_size:(s + 1) * which_batch_size]
        assert (len(labels_selected) == which_batch_size)

        labels_bg, bg_data, frame_num = D.load_data_single(labels_selected, which_labels, frames, which_data='bg',
                                                           resize=True, ordered=ordered, twostream=twostream,
                                                           same_frame=same_frame, trait=trait)
        labels_face, face_data, _ = D.load_data_single(labels_selected, which_labels, frames, which_data='face',
                                                       resize=True, ordered=ordered, twostream=twostream,
                                                       frame_num=frame_num, same_frame=same_frame, trait=trait)

        if C.ON_GPU:
            bg_data = to_gpu(bg_data, device=C.DEVICE)
            face_data = to_gpu(face_data, device=C.DEVICE)
            labels = to_gpu(labels_bg, device=C.DEVICE)

        with cp.cuda.Device(C.DEVICE):
            if which == 'train':
                config = True
            else:
                config = False

            with chainer.using_config('train', False):
                prediction_bg, bg_activations = bg_model(bg_data)
                prediction_face, face_activations = face_model(face_data)

            with chainer.using_config('train', config):
                if config:
                    model.cleargrads()
                prediction = model(bg_activations, face_activations)

                loss = mean_absolute_error(prediction, labels)

                if which == 'train':
                    loss.backward()
                    optimizer.update()

        if record_loss:
            loss_tmp.append(float(loss.data))
            pd_tmp[s] = U.pred_diff_trait(to_cpu(prediction.data), to_cpu(labels))
        if record_predictions and which == 'test':
            preds[s] = to_cpu(prediction.data)

    if record_loss:
        pred_diff[0] = np.mean(pd_tmp, axis=0)
        loss_tmp_mean = np.mean(loss_tmp, axis=0)
        loss_saving.append(loss_tmp_mean)
        print('E %d. %s loss: ' %(0, which), loss_tmp_mean,
              ' pred diff %s: ' % trait, pred_diff[0],
              ' time: ', time.time() - ts)

        U.record_loss_sanity(which, loss_tmp_mean, pred_diff[0])

        if which == 'test' and save_all_results:
            U.record_loss_all_test(loss_tmp, trait=True)

    if record_predictions and which == 'test':
        U.record_all_predictions(which, preds)


def main_loop(which):
    model_number = 127  # TODO: check this!!!!!!!!!! OCE: 105 AS: 61
    index = 0
    #          0    1    2    3    4
    traits = ['O', 'C', 'E', 'A', 'S']
    which_trait = traits[index]

    bgs = ['epoch_89_60_O', 'epoch_79_60_C', 'epoch_99_60_E', 'epoch_89_60_A', 'epoch_89_60_S']
    faces = ['epoch_39_59_O', 'epoch_19_59_C', 'epoch_99_59_E', 'epoch_89_59_A', 'epoch_19_59_S']
    # bg_and_face = ['epoch_99_105_O', 'epoch_99_105_C', 'epoch_19_105_E', 'epoch_99_61_A', 'epoch_9_61_S'] # no regularized
    bg_and_face = ['epoch_79_127_O', 'epoch_19_127_C', 'epoch_39_127_E', '', 'epoch_29_127_S'] # regularized




    bg_name = bgs[index]
    face_name = faces[index]

    if which == 'val':
        saved_epochs = [9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
        models_to_load = ['epoch_%d_%d_%s' % (saved_epochs[i], model_number, which_trait) for i in range(len(saved_epochs))]
    else:
        models_to_load = [bg_and_face[index]]

    for i, model_name in enumerate(models_to_load):
        my_model, bg_model, face_model, my_optimizer, epochs, labels, steps, loss, pred_diff, id_frames = \
            initialize(which, model_name, bg_name, face_name)

        if which == 'val':
            run(which=which, steps=steps, which_labels=labels, frames=id_frames, model=my_model, bg_model=bg_model,
                face_model=face_model, optimizer=my_optimizer, pred_diff=pred_diff, loss_saving=loss, trait=which_trait,
                ordered=True, save_all_results=False, twostream=False, same_frame=True, record_loss=True,
                record_predictions=False)
        elif which == 'test':
            run(which='test', steps=steps, which_labels=labels, frames=id_frames, model=my_model, bg_model=bg_model,
                face_model=face_model, optimizer=my_optimizer, pred_diff=pred_diff, loss_saving=loss, trait=which_trait,
                ordered=True, twostream=False, save_all_results=True, same_frame=True, record_loss=True,
                record_predictions=True)  # ordered=True so will not shuffle


main_loop('test')

'''
RESULTS 

before:
best val 'all', no decay: epoch_79_61_O, epoch_89_61_C, epoch_69_61_E, epoch_29_61_A, epoch_29_61_S

best val 'bg': epoch_59_60_O, epoch_79_60_C, epoch_89_60_E, epoch_89_60_A, epoch_89_60_S
best val 'face': epoch_39_59_O, epoch_49_59_C, epoch_99_59_E, epoch_89_59_A, epoch_19_59_S

best val 'all', with decay 0.001: epoch_69_95_O, epoch_99_95_C, epoch_29_95_E, epoch_19_95_A, epoch_59_95_S

as of 1 apr 19:

best val 'all': epoch_99_105_O, epoch_99_105_C, epoch_19_105_E, epoch_99_61_A, epoch_9_61_S

best val 'bg': epoch_89_60_O, epoch_79_60_C, epoch_99_60_E, epoch_89_60_A, epoch_89_60_S
best val 'face': epoch_39_59_O, epoch_19_59_C, epoch_99_59_E, epoch_89_59_A, epoch_19_59_S



'''


