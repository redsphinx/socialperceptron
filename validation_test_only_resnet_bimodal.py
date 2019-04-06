import numpy as np
import h5py as h5
import time
import os
from tqdm import tqdm

import deepimpression2.constants as C
import deepimpression2.paths as P
import deepimpression2.chalearn30.data_utils as D
import deepimpression2.util as U
from deepimpression2.model_resnet18 import ResNet18_LastLayers, hackyResNet18

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
        face_model = hackyResNet18(num_traits, pretrain)
    else:
        face_model = resnet18()
        face_model.fc = nn.Linear(in_features=512, out_features=num_traits, bias=True)

    p = os.path.join(P.MODELS, 'epoch_14_128')
    face_model.load_state_dict(torch.load(p))
    face_model.fc = nn.Sequential()

    if model_number in hacky_models:
        bg_model = hackyResNet18(num_traits, pretrain)
    else:
        bg_model = resnet18()
        bg_model.fc = nn.Linear(in_features=512, out_features=num_traits, bias=True)

    p = os.path.join(P.MODELS, 'epoch_49_129')
    bg_model.load_state_dict(torch.load(p))
    bg_model.fc = nn.Sequential()

    my_model = ResNet18_LastLayers(num_traits)

    if load_model:
        p = os.path.join(P.MODELS, model_name)
        my_model.load_state_dict(torch.load(p))
        print('model %s loaded' % model_name)

    face_model.cuda(device)
    bg_model.cuda(device)
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

    return my_model, face_model, bg_model, labels, steps, loss, pred_diff, id_frames, loss_function, device, num_traits


def run(which, steps, which_labels, frames, model, face_model, bg_model, pred_diff, loss_saving, trait, ordered,
        save_all_results, record_predictions, record_loss, is_resnet18, num_traits, device, loss_function,
        resnet18_pretrain):
    print('steps: ', steps)
    assert (which in ['test', 'val'])
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

        labels_bg, bg_data, frame_num = D.load_data(labels_selected, which_labels, frames, which_data='bg',
                                                    ordered=True, is_resnet18=is_resnet18, same_frame=True,
                                                    resize=True, resnet18_pretrain=resnet18_pretrain)
        labels_face, face_data, _ = D.load_data(labels_selected, which_labels, frames, which_data='face',
                                                ordered=True, is_resnet18=is_resnet18, same_frame=True,
                                                frame_num=frame_num, resize=True, resnet18_pretrain=resnet18_pretrain)

        if C.ON_GPU:
            bg_data = torch.from_numpy(bg_data)
            bg_data = bg_data.cuda(device)

            face_data = torch.from_numpy(face_data)
            face_data = face_data.cuda(device)

            labels = torch.from_numpy(labels_bg)
            labels = labels.cuda(device)

        model.eval()
        bg_model.eval()
        face_model.eval()
        with torch.no_grad():
            bg_activations = bg_model(bg_data)
            face_activations = face_model(face_data)
            predictions = model(bg_activations, face_activations)
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


def main_loop(which):
    which_trait = None
    PRETRAIN = True

    hacky_models = [146, 145]

    model_number = 145

    if which == 'val':
        if model_number in hacky_models:
            # saved_epochs = [4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 84, 89, 94, 99]
            saved_epochs = [29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 84, 89, 94, 99]
        else:
            saved_epochs = [9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
        models_to_load = ['epoch_%d_%d' % (saved_epochs[i], model_number) for i in range(len(saved_epochs))]
    else:
        models_to_load = ['epoch_19_145']

    for i, model_name in enumerate(models_to_load):
        my_model, face_model, bg_model, labels, steps, loss, pred_diff, id_frames, loss_function, device, num_traits = \
            initialize(which, model_name, PRETRAIN, model_number, hacky_models)

        if which == 'val':
            run(which=which, steps=steps, which_labels=labels, frames=id_frames, model=my_model, bg_model=bg_model,
                face_model=face_model, pred_diff=pred_diff,
                loss_saving=loss, trait=which_trait, ordered=True,
                save_all_results=False, record_predictions=False, record_loss=True, is_resnet18=True,
                num_traits=num_traits, device=device, loss_function=loss_function, resnet18_pretrain=PRETRAIN)

        elif which == 'test':
            run(which=which, steps=steps, which_labels=labels, frames=id_frames, model=my_model, bg_model=bg_model,
                face_model=face_model, pred_diff=pred_diff,
                loss_saving=loss, trait=which_trait, ordered=True,
                save_all_results=True, record_predictions=True, record_loss=True, is_resnet18=True,
                num_traits=num_traits, device=device, loss_function=loss_function, resnet18_pretrain=PRETRAIN)


main_loop('test')

'''
RESULTS

best val 'bg': epoch_99_102
best val 'face': epoch_99_101
best val 'all': epoch_99_113

'''
