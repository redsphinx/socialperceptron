import numpy as np
import os
import h5py as h5
from random import shuffle
import time
from tqdm import tqdm

from torchvision.models import resnet18
from torch.optim.adam import Adam
from torch.optim.sgd import SGD
from torch.optim import lr_scheduler
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, L1Loss
from torch.nn import Tanh

import deepimpression2.paths as P
import deepimpression2.constants as C
import deepimpression2.chalearn20.constants as C2
import deepimpression2.util as U
import deepimpression2.chalearn30.data_utils as D
from deepimpression2.model_resnet18 import hackyResNet18


# settings
# TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
PRETRAIN = True
mode = 'finetune'  # or extractor
num_traits = 5
load_model = False

# device
if C.ON_GPU:
    _dev = 'cuda:%d' % C.DEVICE
else:
    _dev = 'cpu'
device = torch.device(_dev)

# ------------------------------------------------------------------------------
# if mode == 'finetune':
#     # get model
#     my_model = resnet18(pretrained=PRETRAIN)
#     my_model.fc = nn.Linear(in_features=512, out_features=num_traits, bias=True)
#     if load_model:
#         p = os.path.join(P.MODELS, '')
#         my_model.load_state_dict(torch.load(p))
#         print('model  loaded')
#         continuefrom = 0
#     else:
#         continuefrom = 0
#     my_model.cuda(device)
#
#     # https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# elif mode == 'extractor':
#     my_model = resnet18(pretrained=PRETRAIN)
#     for param in my_model.parameters():
#         param.requires_grad = False
#     my_model.fc = nn.Linear(in_features=512, out_features=num_traits, bias=True)
#     my_model = my_model.to(device)
# else:
#     print('mode is not correct: %s' % mode)
#     my_model = None
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# EXPERIMENTAL
m1 = resnet18(pretrained=PRETRAIN)
m1.fc = nn.Sequential()
my_model = hackyResNet18(num_traits)
m1 = m1.to(device)
my_model = my_model.to(device)
# ------------------------------------------------------------------------------


# which_opt = 'sgd'
which_opt = 'adam'

if which_opt == 'sgd':
    learning_rate = 0.001
    momentum = 0.9
    if mode == 'finetune':
        # EXPERIMENTAL
        my_optimizer = SGD(list(m1.parameters()) + list(my_model.parameters()), lr=learning_rate, momentum=momentum)
        # my_optimizer = SGD(my_model.parameters(), lr=learning_rate, momentum=momentum)
    elif mode == 'extractor':
        my_optimizer = SGD(my_model.fc.parameters(), lr=learning_rate, momentum=momentum)
    else:
        print('problem with mode', mode)
        my_optimizer = None
    # Decay LR by a factor of 0.1 every 7 epochs
    # exp_lr_scheduler = lr_scheduler.StepLR(my_optimizer, step_size=7, gamma=0.1)

elif which_opt == 'adam':
    learning_rate = 0.0002
    betas = (0.5, 0.999)
    eps=10-8

    if mode == 'finetune':
        my_optimizer = Adam(list(m1.parameters()) + list(my_model.parameters()), lr=learning_rate, betas=betas, eps=eps, weight_decay=0.001)
        # my_optimizer = Adam(my_model.parameters(), lr=learning_rate, betas=betas, eps=eps, weight_decay=0.001)
    elif mode == 'extractor':
        my_optimizer = Adam(my_model.fc.parameters(), lr=learning_rate, betas=betas, eps=eps)

loss_function = L1Loss()

print('Initializing')
print('model initialized with %d parameters' % (U.get_torch_params(my_model) + U.get_torch_params(m1)))

# train model
epochs = C.EPOCHS
# epochs = 1

train_labels = h5.File(P.CHALEARN_TRAIN_LABELS_20, 'r')
train_loss = []
pred_diff_train = np.zeros((epochs, num_traits), float)
# training_steps = len(train_labels) // C.TRAIN_BATCH_SIZE
training_steps = 100

id_frames = h5.File(P.NUM_FRAMES, 'r')


def run(which, steps, which_labels, frames, first_part, model, optimizer, pred_diff, loss_saving, which_data, trait=None,
        ordered=False, save_all_results=False, record_predictions=False, record_loss=True, is_resnet18=True,
        pretrain_resnet=PRETRAIN):
    print('steps: ', steps)
    assert(which in ['train'])
    assert(which_data in ['bg', 'face'])
    if trait is not None:
        assert(trait in ['O', 'C', 'E', 'A', 'S'])

    if which == 'train':
        which_batch_size = C.TRAIN_BATCH_SIZE

    loss_tmp = []
    pd_tmp = np.zeros((steps, num_traits), dtype=float)
    _labs = list(which_labels)

    preds = np.zeros((steps, num_traits), dtype=float)

    if not ordered:
        shuffle(_labs)

    ts = time.time()
    for s in tqdm(range(steps)):
        labels_selected = _labs[s * which_batch_size:(s + 1) * which_batch_size]
        assert (len(labels_selected) == which_batch_size)
        labels, data, _ = D.load_data(labels_selected, which_labels, frames, which_data, ordered=ordered,
                                      is_resnet18=is_resnet18, resize=True, resnet18_pretrain=pretrain_resnet)

        if C.ON_GPU:
            data = torch.from_numpy(data)
            data = data.cuda(device)
            labels = torch.from_numpy(labels)
            labels = labels.cuda(device)

        # exp_lr_scheduler.step()

        first_part.train()
        model.train()
        optimizer.zero_grad()
        # with torch.set_grad_enabled(True):
        features = first_part(data)
        predictions = model(features)
        # predictions = model(data)
        loss = loss_function(predictions, labels)
        loss.backward()
        optimizer.step()
        if bool(torch.isnan(loss)):
            print('its happening')

        if record_loss:
            loss_tmp.append(float(loss.data))
            pd_tmp[s] = U.pred_diff_trait(np.array(predictions.cpu().data), np.array(labels.cpu().data))
        if record_predictions and which == 'test':
            preds[s] = np.array(predictions.cpu().data)

    if record_loss:
        pred_diff[e] = np.mean(pd_tmp, axis=0)
        loss_tmp_mean = np.mean(loss_tmp, axis=0)
        loss_saving.append(loss_tmp_mean)
        print('E %d. %s loss: ' %(e, which), loss_tmp_mean,
              ' pred diff %s: ' % trait, pred_diff[e],
              ' time: ', time.time() - ts)

        # U.record_loss_sanity(which, loss_tmp_mean, pred_diff[e])

        if which == 'test' and save_all_results:
            U.record_loss_all_test(loss_tmp, trait=True)

    if record_predictions and which == 'test':
        U.record_all_predictions(which, preds)


print('Enter training loop with validation')
for e in range(0, epochs):
    which_trait = None
    train_on = 'face'
    # ----------------------------------------------------------------------------
    # training
    # ----------------------------------------------------------------------------
    run(which='train', steps=training_steps, which_labels=train_labels, frames=id_frames, first_part=m1,
        model=my_model, optimizer=my_optimizer, pred_diff=pred_diff_train,
        loss_saving=train_loss, which_data=train_on, trait=which_trait)

    # save model
    # if ((e + 1) % 10) == 0:
    #     name = os.path.join(P.MODELS, 'epoch_%d_12' % e)
    #     torch.save(my_model.state_dict(), name)
