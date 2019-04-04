import numpy as np
import h5py as h5
import time
import os
from tqdm import tqdm

import deepimpression2.constants as C
import deepimpression2.paths as P
import deepimpression2.chalearn30.data_utils as D
import deepimpression2.util as U
from deepimpression2.model_resnet18 import hackyResNet18

from torchvision.models import resnet18
import torch
from torch import nn
from torch.nn import L1Loss


def initialize(which, model_name, pretrain, model_number, hacky_models):
    num_traits = 5
    load_model = True

    if C.ON_GPU:
        _dev = 'cuda:%d' % C.DEVICE
    else:
        _dev = 'cpu'
    device = torch.device(_dev)

    if model_number in hacky_models:
        my_model = hackyResNet18(num_traits, pretrain)
    else:
        my_model = resnet18(pretrained=pretrain)
        my_model.fc = nn.Linear(in_features=512, out_features=num_traits, bias=True)

    if load_model:
        p = os.path.join(P.MODELS, model_name)
        my_model.load_state_dict(torch.load(p))
        print('model %s loaded' % model_name)

    my_model.cuda(device)

    loss_function = L1Loss()

    print('Initializing')
    print('model initialized with %d parameters' % U.get_torch_params(my_model))

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
    pred_diff = np.zeros((1, num_traits), float)

    id_frames = h5.File(P.NUM_FRAMES, 'r')

    return my_model, labels, steps, loss, pred_diff, id_frames, loss_function, device, num_traits


def run(which, steps, which_labels, frames, model, pred_diff, loss_saving, which_data, trait, ordered,
        save_all_results, record_predictions, record_loss, is_resnet18, num_traits, device, loss_function,
        resnet18_pretrain):
    print('steps: ', steps)
    assert (which in ['test', 'val'])
    assert (which_data in ['bg', 'face'])
    if trait is not None:
        assert (trait in ['O', 'C', 'E', 'A', 'S'])

    if which == 'val':
        which_batch_size = C.VAL_BATCH_SIZE
    elif which == 'test':
        which_batch_size = C.TEST_BATCH_SIZE

    loss_tmp = []
    pd_tmp = np.zeros((steps, num_traits), dtype=float)
    _labs = list(which_labels)

    preds = np.zeros((steps, num_traits), dtype=float)

    ts = time.time()
    for s in tqdm(range(steps)):
        labels_selected = _labs[s * which_batch_size:(s + 1) * which_batch_size]
        assert (len(labels_selected) == which_batch_size)

        labels, data, _ = D.load_data(labels_selected, which_labels, frames, which_data, ordered=ordered,
                                      is_resnet18=is_resnet18, resnet18_pretrain=resnet18_pretrain, resize=True)

        if C.ON_GPU:
            data = torch.from_numpy(data)
            data = data.cuda(device)
            labels = torch.from_numpy(labels)
            labels = labels.cuda(device)

        model.eval()
        with torch.no_grad():
            predictions = model(data)
            loss = loss_function(predictions, labels)
            loss = loss.detach()

        if record_loss:
            loss_tmp.append(float(loss.data))
            pd_tmp[s] = U.pred_diff_trait(np.array(predictions.cpu().data), np.array(labels.cpu().data))
        if record_predictions and which == 'test':
            preds[s] = np.array(predictions.cpu().data)

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
    which_trait = None
    # for normalization of images
    PRETRAIN = True
    hacky_models = [131, 130, 129]

    model_number = 129

    # if val_test_on == 'face':
    #     model_number = 101
    # elif val_test_on == 'bg':
    #     model_number = 102
    # else:
    #     print('val_test_on is not correct')
    #     model_number = None

    if which == 'val':
        if model_number in hacky_models:
            saved_epochs = [4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 84, 89, 94, 99]
        else:
            saved_epochs = [9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
        models_to_load = ['epoch_%d_%d' % (saved_epochs[i], model_number) for i in range(len(saved_epochs))]
    else:
        models_to_load = ['epoch_14_131']

    for i, model_name in enumerate(models_to_load):
        my_model, labels, steps, loss, pred_diff, id_frames, loss_function, device, num_traits = \
            initialize(which, model_name, model_number=model_number, pretrain=PRETRAIN, hacky_models=hacky_models)

        if which == 'val':
            run(which=which, steps=steps, which_labels=labels, frames=id_frames, model=my_model, pred_diff=pred_diff,
                loss_saving=loss, which_data=val_test_on, trait=which_trait, ordered=True,
                save_all_results=False, record_predictions=False, record_loss=True, is_resnet18=True,
                num_traits=num_traits, device=device, loss_function=loss_function, resnet18_pretrain=PRETRAIN)

        elif which == 'test':
            run(which=which, steps=steps, which_labels=labels, frames=id_frames, model=my_model, pred_diff=pred_diff,
                loss_saving=loss, which_data=val_test_on, trait=which_trait, ordered=True,
                save_all_results=True, record_predictions=True, record_loss=True, is_resnet18=True,
                num_traits=num_traits, device=device, loss_function=loss_function, resnet18_pretrain=PRETRAIN)


main_loop('val', 'bg')

'''
RESULTS

best val 'bg': epoch_99_102
best val 'face': epoch_99_101

'''
