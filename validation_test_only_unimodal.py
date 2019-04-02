# TODO: write loop to validate train_59 and train_60 every 10 epochs
import chainer
import numpy as np
from deepimpression2.model_59 import Deepimpression
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


def initialize(which, model_name):
    my_model = Deepimpression()

    load_model = True
    if load_model:
        p = os.path.join(P.MODELS, model_name)
        chainer.serializers.load_npz(p, my_model)
        print('%s loaded' % model_name)

    my_optimizer = Adam(alpha=0.0002, beta1=0.5, beta2=0.999, eps=10e-8)
    my_optimizer.setup(my_model)

    if C.ON_GPU:
        my_model = my_model.to_gpu(device=C.DEVICE)

    print('Initializing')
    print('model initialized with %d parameters' % my_model.count_params())

    # epochs = C.EPOCHS
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

    return my_model, my_optimizer, epochs, labels, steps, loss, pred_diff, id_frames


def run(which, steps, which_labels, frames, model, optimizer, pred_diff, loss_saving, which_data, trait, ordered,
        save_all_results, record_predictions, record_loss):
    print('steps: ', steps)
    assert (which in ['train', 'test', 'val'])
    assert (which_data in ['bg', 'face'])
    assert (trait in ['O', 'C', 'E', 'A', 'S'])

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
        labels, data, _ = D.load_data_single(labels_selected, which_labels, frames, which_data, resize=True,
                                             ordered=ordered, trait=trait)

        if C.ON_GPU:
            data = to_gpu(data, device=C.DEVICE)
            labels = to_gpu(labels, device=C.DEVICE)

        with cp.cuda.Device(C.DEVICE):
            if which == 'train':
                config = True
            else:
                config = False

            with chainer.using_config('train', config):
                if which == 'train':
                    model.cleargrads()
                prediction, _ = model(data)

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
        print('E %d. %s loss: ' % (0, which), loss_tmp_mean,
              ' pred diff %s: ' % trait, pred_diff[0],
              ' time: ', time.time() - ts)

        U.record_loss_sanity(which, loss_tmp_mean, pred_diff[0])

        if which == 'test' and save_all_results:
            U.record_loss_all_test(loss_tmp, trait=True)

    if record_predictions and which == 'test':
        U.record_all_predictions(which, preds)


def main_loop(which, val_test_on):
    if val_test_on == 'face':
        model_number = 59
    elif val_test_on == 'bg':
        model_number = 60
    else:
        print('val_test_on is not correct')
        model_number = None

    if which == 'val':
        saved_epochs = [9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
        # which_trait = 'O'
        # which_trait = 'C'
        # which_trait = 'E'
        # which_trait = 'A'
        which_trait = 'S'
        models_to_load = ['epoch_%d_%d_%s' % (saved_epochs[i], model_number, which_trait) for i in range(len(saved_epochs))]
    else:
        # TODO: for test change here!!!!!!!!!!!!
        which_trait = 'S'
        models_to_load = ['epoch_89_60_S']

    for i, model_name in enumerate(models_to_load):
        my_model, my_optimizer, epochs, labels, steps, loss, pred_diff, id_frames = initialize(which, model_name)

        if which == 'val':
            run(which=which, steps=steps, which_labels=labels, frames=id_frames,
                model=my_model, optimizer=my_optimizer, pred_diff=pred_diff,
                loss_saving=loss, which_data=val_test_on, trait=which_trait, ordered=True, record_loss=True,
                record_predictions=False, save_all_results=False)
        elif which == 'test':
            run(which=which, steps=steps, which_labels=labels, frames=id_frames,
                model=my_model, optimizer=my_optimizer, pred_diff=pred_diff,
                loss_saving=loss, which_data=val_test_on, ordered=True, save_all_results=True,
                trait=which_trait, record_loss=True, record_predictions=True)


main_loop('test', 'bg')

'''
RESULTS

before:
best val 'bg': epoch_59_60_O, epoch_79_60_C, epoch_89_60_E, epoch_89_60_A, epoch_89_60_S
best val 'face': epoch_39_59_O, epoch_49_59_C, epoch_99_59_E, epoch_89_59_A, epoch_19_59_S

as of 1 apr 2019:
best val 'bg': epoch_89_60_O, epoch_79_60_C, epoch_99_60_E, epoch_89_60_A, epoch_89_60_S
best val 'face': epoch_39_59_O, epoch_19_59_C, epoch_99_59_E, epoch_89_59_A, epoch_19_59_S

'''

